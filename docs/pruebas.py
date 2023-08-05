import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generar datos de ejemplo
np.random.seed(42)
n_steps = 100
time = np.linspace(0, 20, n_steps)
sales = 3 * np.sin(time) + np.random.randn(n_steps)

# Crear la serie temporal
data = pd.Series(sales, index=pd.date_range(start='2023-01-01', periods=n_steps, freq='D'))

# Ajustar el modelo ARIMA
order = (2, 1, 1)  # (p, d, q)
model_arima = sm.tsa.ARIMA(data, order=order)
results_arima = model_arima.fit()

""" # Ajustar el modelo SARIMA
order = (2, 1, 1)  # (p, d, q)
seasonal_order = (1, 0, 1, 7)  # (P, D, Q, S)
model_sarima = sm.tsa.SARIMA(data, order=order, seasonal_order=seasonal_order)
results_sarima = model_sarima.fit() """

# Realizar predicciones
start_date = '2023-04-01'
end_date = '2023-04-10'
forecast_arima = results_arima.predict(start=start_date, end=end_date)
# forecast_sarima = results_sarima.predict(start=start_date, end=end_date)

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.plot(data, label='Datos reales')
plt.plot(forecast_arima, label='Predicciones ARIMA')
# plt.plot(forecast_sarima, label='Predicciones SARIMA')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.title('Predicciones ARIMA y SARIMA')
plt.show()