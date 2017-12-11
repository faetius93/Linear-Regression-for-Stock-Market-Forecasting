# STOCK MARKET FORECASTING
# This simple python programme show a math approach to some financial issue.
# Dataset analysis is performed using that techniques:
# - Linear Regression

import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

dates = []
prices_open = []
prices_close = []
prices_adj_close = []
prices_high = []
prices_low = []

def draw_real_prediction():
	# The line / model
	plt.scatter(prices_test, prices_prediction)
	plt.title('Real Values VS Predict Values')
	plt.xlabel('Real Values')
	plt.ylabel('Predict Values')
	plt.grid(axis='y', linestyle='dashed')
	plt.savefig('RealValues_VS_PredictValues01')
	plt.show()

def draw_regression():
	# Plot outputs
	plt.scatter(dates_test, prices_test, color='black', alpha=0.8, s=1.5)
	plt.plot(dates_test, prices_prediction, color='red', ls='dashed', lw=3.5, label= 'Linear model')
	plt.title('Linear Regression')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.grid(axis='y', linestyle='dashed')
	plt.legend()
	plt.savefig('forecasting01')
	plt.show()

# ingestion dei dati storici
with open('ACN.csv', 'r') as csvfile:
	csvFileReader = csv.reader(csvfile)
	next(csvFileReader) # skipping column names
	for row in csvFileReader:
		utc_time = datetime.strptime(row[0][:10], '%Y-%m-%d').timestamp()
		dates.append(utc_time)
		prices_open.append(float(row[1]))
		prices_high.append(float(row[2]))
		prices_low.append(float(row[3]))
		prices_close.append(float(row[4]))
		prices_adj_close.append(float(row[5]))
		
# Analisi dei dati: Varianza, Media, Deviazione Standard
print('\nPrice Open - Media: ', np.mean(prices_open))
print('Price Open - Varianza: ', np.var(prices_open))
print('Price Open - Deviazione Standard: ', np.std(prices_open))

print('\nprices high - Media: ', np.mean(prices_high))
print('prices high - Varianza: ', np.var(prices_high))
print('prices high - Deviazione Standard: ', np.std(prices_high))

print('\nprices low - Media: ', np.mean(prices_low))
print('prices low - Varianza: ', np.var(prices_low))
print('prices low - Deviazione Standard: ', np.std(prices_low))

print('\nprices close - Media: ', np.mean(prices_close))
print('prices close - Varianza: ', np.var(prices_close))
print('prices close - Deviazione Standard: ', np.std(prices_close))

print('\nprices adj close - Media: ', np.mean(prices_adj_close))
print('prices adj close - Varianza: ', np.var(prices_adj_close))
print('prices adj close - Deviazione Standard: ', np.std(prices_adj_close))

# ******************************************************************
# ******************************************************************
#		
# Verifica delle ipotesi di applicabilità della retta di regressione
#
# ******************************************************************
# ******************************************************************

# 1 - coefficiente di correlazione di Pearson
pearson_correlation_coefficients_open = np.corrcoef(dates,prices_open)[0,1]
pearson_correlation_coefficients_close = np.corrcoef(dates,prices_close)[0,1]
pearson_correlation_coefficients_high = np.corrcoef(dates,prices_high)[0,1]
pearson_correlation_coefficients_low = np.corrcoef(dates,prices_low)[0,1]
pearson_correlation_coefficients_adj_close = np.corrcoef(dates,prices_adj_close)[0,1]
print('\nPerason Correlation Coefficients:\n')
print('Data and Price Open: ', pearson_correlation_coefficients_open)
print('Data and Price Close: ', pearson_correlation_coefficients_close)
print('Data and Price High: ', pearson_correlation_coefficients_high)
print('Data and Price Low: ', pearson_correlation_coefficients_low)
print('Data and Price Adj_close: ', pearson_correlation_coefficients_adj_close)

# 2 - split del dataset
dates_train, dates_test, prices_train, prices_test = train_test_split(dates, prices_adj_close, test_size=0.4, random_state=0)

dates_train = np.reshape(dates_train,(len(dates_train),1))
dates_test = np.reshape(dates_test,(len(dates_test),1))
prices_train = np.reshape(prices_train,(len(prices_train),1))
prices_test = np.reshape(prices_test,(len(prices_test),1))

# Define the Linear Model
linear_mod = linear_model.LinearRegression()
linear_mod.fit(dates_train, prices_train)

prices_prediction = linear_mod.predict(dates_test)

# The line / model
#plt.scatter(prices_test, prices_prediction)
#plt.title('Real Values VS Predict Values')
#plt.xlabel('Real Values')
#plt.ylabel('Predict Values')
#plt.grid(axis='y', linestyle='dashed')
#plt.savefig('RealValues_VS_PredictValues01')
#plt.show()

# The coefficients
print('\n')
print('Coefficiente: ', linear_mod.coef_)
print('Intercetta: ', linear_mod.intercept_)
# The mean squared error
print("Errore quadratico medio: %.2f" % mean_squared_error(prices_test, prices_prediction))
# Explained variance score: 1 is perfect prediction
print('Varianza: %.2f' % r2_score(prices_test, prices_prediction))

choice ='1'
while(choice != '0'):
	# Simple Menu:
	print('Seleziona il grafico da visualizzare:')
	print('1 - Correlazione bivariata Dati Stimati VS Dati Reali')
	print('2 - Grafico di Regressione Lineare')
	print('0 - Termina esecuzione')
	
	choice = input("\nPlease make a choice: ")
	
	if choice == "1":
		draw_real_prediction()
	elif choice == "2":
		draw_regression()
	elif choice == "0":
		quit()
	else:
		print("I don't understand your choice.\n\n")
		main()