import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math, tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.2f}'.format

try:
    df = pd.read_csv("/content/All24.csv", sep=";")
except FileNotFoundError:
    print("WARNING: File /content/All24.csv tidak ditemukan.")
    print("Membuat data dummy untuk menjalankan sisa script...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    products = ['Produk A', 'Produk B', 'Produk C']
    data_list = []
    for prod in products:
        for date in dates:
            if date.dayofweek < 6:
                data_list.append({
                    'Tanggal': date.strftime('%d/%m/%Y'),
                    'Produk': prod,
                    'Qty Laku': np.random.randint(5, 50)
                })
    df = pd.DataFrame(data_list)

df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
df = df.sort_values(['Produk', 'Tanggal'])

def create_dataset(dataset, window=14):
    X, y = [], []
    for i in range(len(dataset) - window):
        X.append(dataset[i:i+window])
        y.append(dataset[i+window, 0])
    return np.array(X), np.array(y)

hasil_prediksi = []
produk_skip = []
detail_prediksi_list = []
produk_list = df['Produk'].unique()

print(f"Memulai pemrosesan untuk {len(produk_list)} produk...")

for produk in produk_list:
    data = df[df['Produk'] == produk].copy()

    data_asli = data.groupby('Tanggal', as_index=False)['Qty Laku'].sum().set_index('Tanggal')
    
    if len(data) < 30:
        produk_skip.append({
            'No': len(produk_skip)+1,
            'Produk': produk,
            'Jumlah Record': len(data)
        })
        # print(f"⏭️ Data {produk} terlalu sedikit ({len(data)} baris), dilewati.")
        continue

    data = data.groupby('Tanggal', as_index=False)['Qty Laku'].sum()
    data = data.set_index('Tanggal').asfreq('D')
    data['Qty Laku'] = data['Qty Laku'].interpolate(method='linear')

    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['week'] = data.index.isocalendar().week.astype(int)
    data['Qty_smooth'] = data['Qty Laku'].rolling(window=3, min_periods=1).mean()

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[['Qty_smooth', 'dayofweek', 'month', 'week']])

    window_size = 14
    X, y = create_dataset(scaled_features, window_size)

    if len(X) < 20:
        produk_skip.append({
            'No': len(produk_skip)+1,
            'Produk': produk,
            'Jumlah Record': len(data)
        })
        # print(f"⏭️ Data {produk} kurang panjang untuk window {window_size}, dilewati.")
        continue

    train_size = int(len(X)*0.85)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])),
        Dropout(0.05),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=1e-5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=4,
        verbose=0,
        callbacks=callbacks
    )

    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test),3))], axis=1)
    )[:,0]
    y_pred_inv = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((len(y_pred),3))], axis=1)
    )[:,0]

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    y_test_inv_safe = y_test_inv.copy()
    y_test_inv_safe[y_test_inv_safe == 0] = 1
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv_safe)) * 100
    akurasi = 100 - mape

    hasil_prediksi.append({
        'Produk': produk,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Akurasi': akurasi,
        'history': history,
        'data': data,
        'y_pred_inv': y_pred_inv,
        'train_size': train_size,
        'window_size': window_size
    })

    test_dates = data.index[train_size+window_size:]

    if len(test_dates) == len(y_pred_inv):

        df_detail_produk = pd.DataFrame({
            'Tanggal': test_dates,
            'NamaProduk': produk,
            'QtyPrediksi': y_pred_inv
        })

        df_detail_produk = pd.merge(
            df_detail_produk,
            data_asli.rename(columns={'Qty Laku': 'QtyJual'}),
            left_on='Tanggal',
            right_index=True,
            how='left'
        )

        df_detail_produk_filtered = df_detail_produk.dropna(subset=['QtyJual'])

        if not df_detail_produk_filtered.empty:
            df_calc = df_detail_produk_filtered.copy()

            df_calc['Selisih Prediksi'] = df_calc['QtyPrediksi'] - df_calc['QtyJual']

            qty_jual_safe = df_calc['QtyJual'].copy()
            qty_jual_safe[qty_jual_safe == 0] = 1

            df_calc['mape'] = (np.abs(df_calc['Selisih Prediksi']) / qty_jual_safe) * 100
            df_calc['mae'] = np.abs(df_calc['Selisih Prediksi'])
            df_calc['mse'] = df_calc['Selisih Prediksi']**2
            df_calc['rmse'] = np.abs(df_calc['Selisih Prediksi'])

            df_calc = df_calc[[
                'Tanggal',
                'NamaProduk',
                'QtyJual',
                'QtyPrediksi',
                'Selisih Prediksi',
                'mape',
                'mae',
                'rmse',
                'mse'
            ]]

            detail_prediksi_list.append(df_calc)

print("...Selesai melakukan training & evaluasi untuk semua produk.")

if hasil_prediksi:
    df_hasil = pd.DataFrame(hasil_prediksi)[['Produk', 'MSE', 'RMSE', 'MAE', 'MAPE', 'Akurasi']]
    df_hasil = df_hasil.sort_values('Akurasi', ascending=False).reset_index(drop=True)
    df_hasil.index += 1
    df_hasil.index.name = "Rangking"

    print("\n📊 RANGKUMAN AKURASI PER PRODUK (OVERALL):")
    print(df_hasil.to_string(index=True, justify='center', col_space=12))

    top_10 = df_hasil.head(10)['Produk'].values
    bottom_10 = df_hasil.tail(10)['Produk'].values

    for kategori, daftar in [('TERBAIK', top_10), ('TERBURUK', bottom_10)]:
        print(f"\n📈 Grafik Prediksi Produk {kategori}:")
        for produk in daftar:
            record = next(item for item in hasil_prediksi if item['Produk'] == produk)

            data = record['data']
            y_pred_inv = record['y_pred_inv']
            train_size = record['train_size']
            window_size = record['window_size']

            test_start_index = train_size + window_size
            if test_start_index < len(data.index):
                test_dates = data.index[test_start_index:]

                if len(test_dates) == len(y_pred_inv):
                    plt.figure(figsize=(12,5))
                    plt.plot(data.index, data['Qty Laku'], color='blue', label='Data Asli')
                    plt.plot(test_dates, y_pred_inv, color='red', label='Hasil Prediksi (Test)')
                    plt.axvline(x=data.index[test_start_index], color='black', linestyle='--', label='Batas Train/Test')
                    plt.title(f"📊 Prediksi Penjualan Harian ({kategori}) - {produk}")
                    plt.xlabel("Tanggal")
                    plt.ylabel("Qty Laku")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                else:
                    print(f"   -> Skipping plot for {produk} (mismatch length)")
            else:
                 print(f"   -> Skipping plot for {produk} (not enough test data)")

else:
    print("\n⚠️ Tidak ada produk yang berhasil diproses.")

if produk_skip:
    df_skip = pd.DataFrame(produk_skip)
    print("\n⚠️ PRODUK YANG TIDAK DIPREDIKSI (DATA KURANG):")
    print(df_skip.to_string(index=False, justify='center', col_space=12))
else:
    print("\n✅ Semua produk memiliki cukup data untuk diproses.")

if detail_prediksi_list:
    df_detail_final = pd.concat(detail_prediksi_list, ignore_index=True)
    df_detail_final['Tanggal'] = df_detail_final['Tanggal'].dt.strftime('%Y-%m-%d')
    df_detail_final.index += 1
    df_detail_final.index.name = "No"

    print("\n\n" + "="*60)
    print("📋 DETAIL PREDIKSI DATA TESTING (METRIK PER BARIS)")
    print("="*60)

    df_display = df_detail_final.copy()
    df_display['QtyJual'] = df_display['QtyJual'].astype(int)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.2f}'.format):
        print(df_display)

    csv_filename = "detail_prediksi_per_baris.csv"
    try:
        df_detail_final.to_csv(csv_filename, index=True, index_label='No', float_format='%.2f', encoding='utf-8')
        print(f"\n✅ Berhasil disimpan ke file CSV: {csv_filename}")
    except Exception as e:
        print(f"\n⚠️ Gagal menyimpan ke CSV: {e}")

else:
    print("\n⚠️ Tidak ada detail prediksi untuk ditampilkan atau disimpan.")
