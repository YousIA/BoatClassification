
import time
import keras
import models
import utils
import parameters
from sklearn.model_selection import train_test_split




#(x_train, y_train ),(x_test, y_test )= keras.datasets.mnist.load_data()
x, y= utils.read_and_resize_dataset(128,128)
x, y=utils.normalize_data(x, y, 128,128, parameters.number_channels,
                                             parameters.num_classes)
x_array_train,x_array_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
cnn_first_model= models.CNN_first_model(parameters.num_classes,
                                        input_shape=(parameters.width, parameters.height, parameters.number_channels)
                                        )
cnn_second_model = models.CNN_second_model(parameters.num_classes,
                                           input_shape=(parameters.width, parameters.height, parameters.number_channels))
cnn_third_model = models.CNN_third_model(parameters.num_classes,
                                        input_shape=(parameters.width, parameters.height, parameters.number_channels))
cnn_fourth_model = models.CNN_fourth_model(parameters.num_classes,
                                           input_shape=(parameters.width, parameters.height, parameters.number_channels))
cnn_fifth_model = models.CNN_fifth_model(parameters.num_classes,
                                         input_shape=(parameters.width, parameters.height, parameters.number_channels))

y_first_model, score_first_model = models.train_and_test(cnn_first_model, x_array_train,
                                                       y_train, x_array_test, y_test,
                                                       parameters.epochs, parameters.batch_size,
                                                       "first_model")

y_second_model, score_second_model = models.train_and_test(cnn_second_model, x_array_train,
                                                       y_train, x_array_test, y_test,
                                                       parameters.epochs, parameters.batch_size,
                                                       "second_model")

y_third_model, score_third_model=models.train_and_test(cnn_third_model, x_array_train,
                                                       y_train, x_array_test, y_test,
                                                       parameters.epochs, parameters.batch_size,
                                                       "third_model"
                                                       )

y_fourth_model, score_fourth_model=models.train_and_test(cnn_fourth_model, x_array_train,
                                                       y_train, x_array_test, y_test,
                                                       parameters.epochs, parameters.batch_size,
                                                         "fourth_model")

y_fifth_model, score_fifth_model=models.train_and_test(cnn_fifth_model, x_array_train,
                                                       y_train, x_array_test, y_test,
                                                       parameters.epochs, parameters.batch_size,
                                                       "fifth_model")

#first_model.evaluate(x_test,y_test)

print("results of the first model")
print(y_first_model)
print("results of the second model")
print(y_second_model)
print("results of the third model")
print(y_third_model)
print("results of the fourth model")
print(y_fourth_model)
print("results of the fifth model")
print(y_fifth_model)
start_time_ensemble = time.time()
y_ensemble_model = models.ensemble(y_first_model, y_second_model, y_fourth_model, y_fourth_model, y_fifth_model)
end_time_ensemble = time.time()
time_ensemble = end_time_ensemble - start_time_ensemble
print("results of the ensemble model")
print(y_ensemble_model)
start_time_pruning = time.time()
y_ensemble_pruning= models.ensemble_pruning(y_first_model, y_fourth_model, y_fifth_model)
end_time_pruning = time.time()
time_pruning = end_time_pruning - start_time_pruning
print("results of the ensemble pruning odel")
print(y_ensemble_pruning)
print("ground Truth")
print(y_test)
print("percentage of corrected images of the ensemble")
percentage_ensemble = models.percentage_corrected_images(y_test[0], y_ensemble_model)
print(percentage_ensemble)
print("percentage of corrected images of the ensemble pruning")
percentage_ensemble_pruning = models.percentage_corrected_images(y_test[0], y_ensemble_pruning)
print(percentage_ensemble_pruning)
print("Inference time ensemble")
print(time_ensemble)
print("Inference time pruning")
print(time_pruning)
