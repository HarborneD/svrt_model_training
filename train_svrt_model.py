import sys
import os

# INITIALISE FRAMEWORK
###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
# framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool


if __name__ == "__main__":
    
    # model_names = ["vgg16", "vgg16_imagenet", "vgg19_imagenet", "inception_v3_imagenet"]#, "inception_resnet_v2_imagenet", "mobilenet_imagenet", "xception_imagenet"]
    # train_data_sizes = [ 1000, 2000 ]
    
    dataset_name = "SVRT Problem 1"
    model_name = "vgg16"
    train_data_size = 2000

    num_train_steps = 80
    learning_rate = 0.0001

    model_save_path_suffix = "" 

    if(len(sys.argv) > 1):
        dataset_name = sys.argv[1]
        
    if(len(sys.argv) > 2):
        model_name = sys.argv[2]
    
    if(len(sys.argv) > 3):
        train_data_size = int(sys.argv[3])

    if(len(sys.argv) > 4):
        num_train_steps = int(sys.argv[4])

    if(len(sys.argv) > 5):
        learning_rate = float(sys.argv[5])

    oputput_file_name = os.path.join("outputs",dataset_name+"_training_outputs.csv")


    num_validation_folds = 5

    proportion_validation_data = 0.2
    proportion_final_test_data = 0.2
    proportion_train_data = 1.0 - (proportion_validation_data + proportion_final_test_data)
    #INITIALISE EXPERIMENT PARAMETERS    
    
    # model_name = "inception_v3_imagenet"
    normalise_data = True
        
    experiment_id="SVRT_"+dataset_name
    output_path=str(experiment_id)+"_results.csv"
    
    model_train_params ={
    "learning_rate": learning_rate
    ,"batch_size":128
    ,"num_train_steps":num_train_steps
    ,"experiment_id":experiment_id
    }


    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)

    ##DATASET
    dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name, load_split_if_available=False, train_ratio=proportion_train_data,validation_ratio=proportion_validation_data,test_ratio=proportion_final_test_data)

    label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

    #TEST DATA
    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = 400,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num test examples: "+str(len(test_x)))

    # #TRAINING DATA SIZE
    # for train_data_size in train_data_sizes:
    #     #MODELS
    #     for model_name in model_names:
    
    
    #CROSS FOLDS
    for cross_fold_i in range(num_validation_folds):
        print("______")
        print("training size: " + str(train_data_size))
        print("Model Name: " + model_name)
        print("Cross Fold: "+str(cross_fold_i))
        print("")
        #LOAD DATA
        #load all train images as model handles batching
        print("load training data")
        print("")
        source = "train"
        train_x, train_y = dataset_tool.GetBatch(batch_size = train_data_size,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
        print("num train examples: "+str(len(train_x)))

        # standardized_train_x = dataset_tool.StandardizeImages(train_x)

        #validate on up to 256 images only
        source = "validation"
        val_x, val_y = dataset_tool.GetBatch(batch_size = 200,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
        print("num validation examples: "+str(len(val_x)))


        


        model_save_path_suffix = "fold_"+str(cross_fold_i)

        #train model
        model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
    
        #LOAD OR TRAIN MODEL
        load_base_model_if_exist = False
        train_model = False
        
        #LOAD MODEL
        model_load_path = framework_tool.model_save_path
        save_best_path = framework_tool.model_save_path+"_best"
        if(load_base_model_if_exist == True and os.path.exists(model_load_path) == True):
            model_instance.LoadModel(model_load_path)
        else:
            train_model = True
        
        if(train_model):
            #OR TRAIN MODEL
            training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y, early_stop=True, save_best_name=save_best_path)
            model_instance.LoadModel(save_best_path)
            eval_stats = model_instance.EvaluateModel(val_x, val_y, model_train_params["batch_size"])
            
        with open(oputput_file_name , "a") as f:
            f.write(",".join([str(train_data_size),model_name,str(cross_fold_i), str(learning_rate), str(training_stats["accuracy_after_training"][1]), str(eval_stats[1])] ) + "\n" )
        
        dataset_tool.CreateNewValidationFold()

    