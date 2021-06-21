# Diving48
Please download RGB data and annotations (cleaned Version 2 updated on 10/30/2020) from [the Diving48 webpage](http://www.svcl.ucsd.edu/projects/resound/dataset.html).

After downloading the data to a `root` path named diving48, make sure that the folder tree looks like:

	diving48
	├── frames
	│   └── OFxuiqI5G44_00247
	│		└── image_000001.jpg
	│		└── image_000002.jpg
   	│		└── ......	
	│   └── OFxuiqI5G44_00248
	│		└── image_000001.jpg
	│		└── image_000002.jpg
	│		└── .....
    ├── Diving48_V2_train.json
    └── Diving48_V2_test.json


Then set the `root` path in the `configs/*.yaml` files to the path to your Diving48 folder.


# FineGym
The [official FineGym dataset webpage](https://sdolivia.github.io/FineGym/) provides the URL of original YouTube videos for downloading.

 The videos are of about 1 hours long, and need to be cropped into segments using the annotations provided. Due to the copyright concerns, we are not able to provide the cropped video segments/extracted frames for direct downloading. Please follow the instructions on the official webpage to conduct the pre-processing.

After finish cropping the video segments and extracted the video frames, please create a `root` folder named `FineGym` and put the processed data into it, so that the folder tree looks like:

	FineGym
	├── frames
	│   └── Z2T9B4qExzk_E_007618_007687_A_0020_0021
	│ 	    	└── image_000001.jpg
	│		└── image_000002.jpg
	│		└── image_000003.jpg
	│		└── ....
	│   └── zNL3kn3UBmg_E_008111_008200_A_0046_0048
	│ 	    	└── image_000001.jpg
	│		└── image_000002.jpg
	│		└── image_000003.jpg
	│		└── ....
	└── scripts
	    └── gym99_train_element_v1.1.txt
	    └── gym99_val_element.txt
	    └── gym288_train_element_v1.1.txt
	    └── gym288_val_element.txt
	

Then set the `root` path in the `configs/*.yaml` files to the path to your FineGym folder.

# Initialization of Weights

S3D weights pretrained on Kinetics400 can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/research/tqn/K400-weights/S3D_K400.pth.tar) (~30.3MB)

Please set the `pretrained_weights_path` in the corresponding `configs/*_first_stage.yaml` files to the path to where the weights are saved.
