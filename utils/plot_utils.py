import os
import numpy as np
from collections import deque
from threading import Thread 
from queue import Queue

class Logger(object):
	'''write something to txt file'''
	def __init__(self, path):
		self.birth_time = datetime.now()
		filepath = os.path.join(path, self.birth_time.strftime('%Y-%m-%d-%H:%M:%S')+'.log')
		self.filepath = filepath
		with open(filepath, 'a') as f:
			f.write(self.birth_time.strftime('%Y-%m-%d %H:%M:%S')+'\n')

	def log(self, string):
		with open(self.filepath, 'a') as f:
			time_stamp = datetime.now() - self.birth_time
			f.write(strfdelta(time_stamp,"{d}-{h:02d}:{m:02d}:{s:02d}")+'\t'+string+'\n')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name='null', fmt=':.4f'):
		self.name = name 
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.local_history = deque([])
		self.local_avg = 0
		self.history = []
		self.dict = {} # save all data values here
		self.save_dict = {} # save mean and std here, for summary table

	def update(self, val, n=1, history=0, step=5):
		self.val = val
		self.sum += val * n
		self.count += n
		if n == 0: return
		self.avg = self.sum / self.count
		if history:
			self.history.append(val)
		if step > 0:
			self.local_history.append(val)
			if len(self.local_history) > step:
				self.local_history.popleft()
			self.local_avg = np.average(self.local_history)


	def dict_update(self, val, key):
		if key in self.dict.keys():
			self.dict[key].append(val)
		else:
			self.dict[key] = [val]

	def print_dict(self, title='IoU', save_data=False):
		"""Print summary, clear self.dict and save mean+std in self.save_dict"""
		total = []
		for key in self.dict.keys():
			val = self.dict[key]
			avg_val = np.average(val)
			len_val = len(val)
			std_val = np.std(val)

			if key in self.save_dict.keys():
				self.save_dict[key].append([avg_val, std_val])
			else:
				self.save_dict[key] = [[avg_val, std_val]]

			print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
				% (key, title, avg_val, title, std_val, len_val))

			total.extend(val)

		self.dict = {}
		avg_total = np.average(total)
		len_total = len(total)
		std_total = np.std(total)
		print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
			% (title, avg_total, title, std_total, len_total))

		if save_data:
			print('Save %s pickle file' % title)
			with open('img/%s.pickle' % title, 'wb') as f:
				pickle.dump(self.save_dict, f)

	def __len__(self):
		return self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)




class PlotterThread():
	def __init__(self, writer):
		self.writer = writer
		self.task_queue = Queue(maxsize=0)
		worker = Thread(target=self.do_work, args=(self.task_queue,))
		worker.setDaemon(True)
		worker.start()

	def do_work(self, q):
		while True:
			content = q.get()
			if content[-1] == 'image':
				self.writer.add_image(*content[:-1])
			elif content[-1] == 'scalar':
				self.writer.add_scalar(*content[:-1])
			elif content[-1] == 'gif':
				self.writer.add_video(*content[:-1])
			else:
				raise ValueError
			q.task_done()

	def add_data(self, name, value, step, data_type='scalar'):
		self.task_queue.put([name, value, step, data_type])

	def __len__(self):
		return self.task_queue.qsize()


