# On the Generalizability of Deep Learning-based Code Completion Across Programming Language Versions

This is the replication package of `On the Generalizability of Deep Learning-based Code Completion Across Programming Language Versions` in which we investigated the capabilities of a state-of-the-art model, CodeT5, to generalize across nine different Java versions.

## CodeT5 model

### Pipeline
* ##### Datasets and tokenizer

    We trained the model using HuggingFace library. HuggingFace allowed us to load the pretraining checkpoint using 
`model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")` without pretraining the model. We loaded also the tokenizer using 
`tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)`. For this reason we did not share the pretraining model dataset and tokenizer
    You can find the finetuning datasets, one for each java version, [here](https://zenodo.org/XXXX) inside `datasets` folder. 


* ##### Hyper Parameter tuning

    We did hyper parameter tuning to find the best model for the finetuning.
    We tested 3 different learning rates [1e-5, 2e-5, 5e-5], training the model for 10k steps and then evaluating it on the eval set. We performed the operation with and without the pretraining phase.
    
    For performing HP tuning you can run:
    ```
    python main_HP_tuning_with_pretraining \
    --model_name=model/pytorch_model.bin \
    --output_dir=./saving_folder \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --epochs 2 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 8 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 42 
    ```
    You can set the learning rate value to test and the script file (main_HP_tuning_with_pretraining.py or main_HP_tuning_without_pretraining.py) for performing HP tuning for the model respectively with and without the pretraining. 
    
    You can evalute all the models to find the best performing one in the following way:
    ```
    python evaluate_eval_set.py \
    --model_name=saving_folder/checkpoint-10000/pytorch_model.bin \
    --tokenizer_name=Salesforce/codet5-base \
    --do_test \
    --train_batch_size 12 \
    --eval_batch_size 8 \
    --encoder_block_size 512 \
    --output_csv "./HP_tuning/predictions.csv" \
    --decoder_block_size 256 \
    --num_beams=1 \
    --seed 42 
    ```
    You must set the correct path of the 10k checkpoint and the output file for saving the predictions.
    You can find the scripts inside the `scripts/HP_tuning` folder.
    
    The results are the following:
    
    | With Pretraining | Learning rate | Accuracy |
    |------------------|---------------|----------|
    | YES              | 1e-5          | 39%      |
    | YES              | 2e-5          | 41%      |
    | YES              | 5e-5          | 43%      |
    | NO               | 1e-5          | 3%       |
    | NO               | 2e-5          | 5%       |
    | NO               | 5e-5          | 8%       |
    
    The best configuration in both cases is the one with learning rate= 5e-5
    We reported the predictions of all the 6 models [here](https://zenodo.org/XXXX) inside the `HP_tuning` folder.

    
* ##### Finetuning

    We finetuned all the models for 15 epochs and then we evaluated the models every 50k steps.
    
    To perform the training you can run:
    ```
    python main_with_pretraining.py \
    --model_name=model/pytorch_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --epochs 15 \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 8 \
    --num_beams 1 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 42  
    ```
    
    You can select main_with_pretraining.py to train the model with the pretraining, and main_without_pretraining.py to training the model from scratch.
    
    We evaluated all checkpoints running:
    ```
    python evaluate_all_eval_set.py \
    --tokenizer_name=Salesforce/codet5-base \
    --do_test \
    --train_batch_size 6 \
    --eval_batch_size 6 \
    --encoder_block_size 512 \
    --output_csv "./predictions_ft_with/predictions_CHECKPOINT.csv" \
    --decoder_block_size 256 \
    --num_beams=1 \
    --checkpoint "<specifiy the checkpoint to evaluate here>" \
    --checkpoint_folder "saved_models" \
    --seed 42 
    ```
    
    All the scripts can be found in `scripts/finetuning` folder.
    
    The best performing model was the last checkpoint for the model with pretraining and the second-last checkpoint for the model without pretraining (we performed the early stopping condition using 10 as number of checkpoints without improvement). We chose to avoid further training, since the model has been trained for 26 days and they were close to convergence.
    You can find the finetuned models [here](https://zenodo.org/XXXX) inside the `finetuned_models` folder, together with the performance of all evaluated checkpoints.
    
    Once we found the best performing model, we evaluated it on all the test set (one for each java version):
    
    ```
    python evaluate_all_test_set.py \
    --tokenizer_name=Salesforce/codet5-base \
    --do_test \
    --train_batch_size 12 \
    --eval_batch_size 8 \
    --encoder_block_size 512 \
    --output_csv "./predictions/predictions_VERSION.csv" \
    --decoder_block_size 256 \
    --num_beams=1 \
    --dataset_folder=dataset_full \
    --best_model "<path to the pytorch_model.bin>" \
    --seed 42 
    ```
    
    We saved the results [here](https://zenodo.org/XXXX) in the `results_finetuning` folder
    
  
* ##### Second finetuning   
  
    We performed a second finetuning, training a model specific for each java version to see whether a small finetuning is enough for drastically improve the performance on a specific java version.
    We trained each model for 5 epochs, running:
    
    ```
    python main_second_finetuning.py \
    --model_name=<path to starting checkpoint> \
    --output_dir=./saved_models_second_finetuning \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --epochs 5 \
    --dataset_folder=dataset_full \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 8 \
    --num_beams 1 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 42 
    ```
    
    We used as starting checkpoint the final checkpoint of the model with and without the finetuning.
    For both the models, we evaluated on the eval set the performance of the model after each epoch of training, chosing the best performing model.
    
    You can do that running:
    
    ```
    python evaluate_all_eval_set_second_finetuning.py \
    --tokenizer_name=Salesforce/codet5-base \
    --do_test \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --encoder_block_size 512 \
    --output_csv "./predictions_second_finetuning_with/predictions_CHECKPOINT.csv" \
    --decoder_block_size 256 \
    --dataset_folder=dataset_full \
    --num_beams=1 \
    --checkpoint_folder "saved_models_second_ft_with" \
    --seed 42 
    ```
    
    We stored all the best model for each Java version [here](https://zenodo.org/XXXX) in the `second_finetuned_models` folder.
    
    We then evaluated all the models on the test set to see whether the performance improved (results are available in the paper).
    For computing the results you can run:
    
    ```
    python evaluate_all_test_set_second_finetuning_v2.py \
    --tokenizer_name=Salesforce/codet5-base \
    --do_test \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --encoder_block_size 512 \
    --output_csv "./predictions_testset_second_finetuning_with/predictions_CHECKPOINT.csv" \
    --decoder_block_size 256 \
    --num_beams=1 \
    --dataset_folder=dataset_full \
    --best_model "results_second_finetuning/with/best_checkpoint.csv" \
    --checkpoint_folder "saved_models_second_ft_with" \
    --seed 42 
    ```
    
    We stored the predictions of each model  [here](https://zenodo.org/XXXX) in the `results_second_finetuning` folder. 
    
* ##### Analysis
   
    We performed the analysis of the new constructs introduced after Java 8, as we described in the paper.
    You can find the script in the `scripts/analysis` folder.
    
    For the analysis you can simply run:
    ```
    python3 find_constructs.py --prediction_folder "results/" --output_folder "constructs". 
    ```
    You can find, for each of the 100 random samples, the ID of the selected instances, together with the number of correct and wrong prediction in the sample. We used this data for creating the boxplot in the paper
    
    This data can be found [here](https://zenodo.org/XXXX) in the `analysis/constructs` folder. 
    
    To assess the impact of the version-specific finetuning, we performed the Fisher's test for the pre-trained and non pre-trained model.
    We reported in the `statistic_analysis` folder the data used for the computing the Fisher's test (in `data_with_pretraining` and `data_without_pretraining` subfolders) together with the script `fisher.R`. We commented in the paper the results for the model without the pretraining. The results for the pretrained model are the following:

We adopted this script for computing all Fisher's tests described in the paper.