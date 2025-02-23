==========================
INPUTS
==========================
-----------------------------------
TO RUN THIS PROGRAM:
-----------------------------------
1. open the directory /OCTA_tortuosity
2. run the python file using 
	python -m model_eval.main



-----------------------------------
To evaluate a model
-----------------------------------
1. put the .keras file in /OCTA_tortuosity/model_eval
2. rename it as model.keras



-----------------------------------
To set up images to evaluate
-----------------------------------
1. Put the original vessel network file in the 
	/OCTA_tortuosity/model_eval/original_images
2. Put the segmented vessels in the 
	/OCTA_tortuosity/model_eval/images
	
	a. Make sure that the folder name is same as its original name in the /OCTA_tortuosity/model_eval/original_images directory
	
	b. Make sure that the directory is as such
	-images
		-{folder name as original image name}
			-vessels_localized_log.csv
			-result
				-non_tortuous
				-tortuous
==============================
OUTPUTS
==============================
-----------------------------------
Confusion Matrix, Metrics
-----------------------------------
You can find those in the 
/OCTA_tortuosity/model_eval/final_csvs

-----------------------------------
Annotated Images
-----------------------------------
You can find these in the 
/OCTA_tortuosity/model_eval/result

The predicted tortuous vessels are highlighted red in the predicted followed by file name, and the ground truths are in the file starting with original file name.
