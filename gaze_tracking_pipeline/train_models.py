import numpy as np
import scipy
import cv2
from os import path
from gaze_tracking_pipeline import gaze_api
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


def process_data():
	path_to_data = '../dataset_collection_tool'
	processed_dataset_x = []
	processed_dataset_y = []
	with open(path.join(path_to_data, 'annotations.tsv'), newline='\n') as metafile:
		for row, content in tqdm(enumerate(metafile.readlines())):
			try:
				content = content.split('\t')
				index = content[0]
				x = content[1]
				y = content[2]
				img = cv2.imread(path.join(path_to_data, f'img-{index}.png'))
				face, frame = gaze_api(img)
			except:
				continue
			processed_dataset_x.append(np.concatenate((face.left_eye.relative_center, face.right_eye.relative_center,
			                                           face.face_position[0].squeeze(),
			                                           face.face_position[1].squeeze())))
			processed_dataset_y.append(np.array([int(x.split('.')[0]), int(y.split('.')[0])]))
		processed_dataset_x = np.array(processed_dataset_x)
		processed_dataset_y = np.array(processed_dataset_y)
		np.save('processed_dataset_x', processed_dataset_x)
		np.save('processed_dataset_y', processed_dataset_y)
		print(len(processed_dataset_y))
		print('cool')


def train_xgb():
	X = np.load('processed_dataset_x.npy')
	Y = np.load('processed_dataset_y.npy')

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
	# xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
	#                           max_depth=5, alpha=10, n_estimators=10)

	multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear')).fit(X_train, y_train)


	preds = multioutputregressor.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	print("RMSE: %f" % (rmse))
	return multioutputregressor



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
	model.add(Dense(7,  activation='relu'))
	model.add(Dense(2, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def train_dense():
	X = np.load('processed_dataset_x.npy')
	Y = np.load('processed_dataset_y.npy')

	estimator = KerasRegressor(build_fn=baseline_model, epochs=250, batch_size=5, verbose=1)
	kfold = KFold(n_splits=10)
	results = cross_val_score(estimator, X, Y, cv=kfold)
	print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == '__main__':
	train_xgb()

