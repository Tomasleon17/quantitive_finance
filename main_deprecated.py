import pandas as pd
import numpy as np
from modules.backend import market_prices

# Portafolio 
tickers = ['IEF', 'SPTL', 'TLT', 'VGLT']
start = '2023-01-01'
end = '2024-12-31'
df_port = market_prices(start_date=start, end_date=end, tickers=tickers)
##print(df_port.columns)
df_port = df_port[['FECHA', 'TICKER', 'EMISOR', 'PRECIO_CIERRE']]
print(df_port.head(5))

# ===== Volatilidad Portafolio =====

# Vector de precios (weights)
weights = [1/len(tickers)] * len(tickers)
vector_weights = np.array(weights)
vectro_weights_t = np.array([weights])



# Pivot Table
df_pivot = pd.pivot_table(
    data=df_port, 
    index='FECHA', 
    columns='TICKER',
    values='PRECIO_CIERRE',
    aggfunc='max'
    )
print('='*100)
print(df_pivot.head(5))

# Retornos
df_ret = df_pivot.pct_change().dropna()
print('='*100)
print(df_ret.head(5))

# Matriz Varianza - Covarianza
m_cov = df_ret.cov()
print('='*100)  
print(m_cov)
matriz_cov = np.array(m_cov.values)

# Varianza del Portafolio
print('='*100) 
print(f'Dimensión del vector de pesos: {vector_weights.shape}')
print(f'Dimensión de la matriz de covarianza: {matriz_cov.shape}')
print(f'Dimensión del vector de weights traspuesto: {vectro_weights_t.shape}')

print('='*100) 
vector_cov = np.dot(matriz_cov, vector_weights)
print(f'Dimensión del vector de covarianza es: {vector_cov.shape}')
print('='*100)
varianza = np.dot(vectro_weights_t, vector_cov)[0]
print(f'La varianza del portafolio es: {varianza}')

# Volatilidad del Portafolio
vol_port = np.sqrt(varianza) * 100
print(f'La volatilidad del portafolio es: {vol_port}%')

# Volatilidad Anualizada
vol_port_1y = vol_port * np.sqrt(252)
print(f'La volatilidad anual del portafolio es: {vol_port_1y}%')