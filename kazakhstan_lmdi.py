import pandas as pd
import numpy as np
import warnings
import io
import sys

# --- 0. Настройка окружения ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Определение вспомогательных функций ---

def clean_column(series):
    '''
    Очищает столбец pandas Series: удаляет '?', пробелы,
    меняет ',' на '.' и преобразует в число.
    '''
    # Убедимся, что это строка, удаляем '?' и пробелы
    cleaned_series = series.astype(str).str.replace('?', '', regex=False).str.replace(' ', '')
    # Меняем запятую на точку для десятичных чисел
    cleaned_series = cleaned_series.str.replace(',', '.')
    # Преобразуем в число, ошибки станут NaT (Not a Number)
    return pd.to_numeric(cleaned_series, errors='coerce')

def lmdi_weight(v_t, v_0):
    '''
    Рассчитывает логарифмический вес (L) для LMDI.
    L(a, b) = (a - b) / (ln(a) - ln(b))
    '''
    # Обработка крайних случаев, чтобы избежать деления на ноль или log(0)
    if v_t == v_0:
        return v_t
    if v_t == 0 or pd.isna(v_t):
        v_t = 1e-9 # очень маленькое число
    if v_0 == 0 or pd.isna(v_0):
        v_0 = 1e-9 # очень маленькое число
        
    return (v_t - v_0) / (np.log(v_t) - np.log(v_0))

def run_lmdi_analysis():
    '''
    Основная функция для загрузки, очистки и проведения LMDI-анализа.
    '''
    try:
        # --- 2. Загрузка и очистка данных ---
        
        # Загружаем CSV, указывая 4-ю строку (индекс 3) как заголовок
        df = pd.read_csv("date_LMDI_ARDL.csv", sep=';', header=3)
        
        # Словарь для выбора и переименования нужных столбцов
        columns_to_use = {
            'Переменная ->': 'Year',
            'Emissions total  (CO2eq) (AR5) Agriculture': 'E_total',
            'Emissions Crops total (kt CO2eq)': 'E_crops',
            'Emissions Livestock total kt (CO2eq)': 'E_livestock',
            'Gross Production Value Agriculture (mln tenge constant prices 2015)': 'A_total',
            'Gross Production Value Crops ( mln tenge constant prices 2015)': 'A_crops',
            'Gross Production Value Livestock (mln tenge constant prices 2015 )': 'A_livestock'
        }
        
        # Выбираем только нужные столбцы и сразу переименовываем
        df = df[columns_to_use.keys()].rename(columns=columns_to_use)
        
        # Очищаем все столбцы, кроме 'Year'
        cols_to_clean = ['E_total', 'E_crops', 'E_livestock', 'A_total', 'A_crops', 'A_livestock']
        for col in cols_to_clean:
            df[col] = clean_column(df[col])
            
        # Очищаем 'Year' отдельно
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Удаляем строки, где год не распознался, и фильтруем по годам (2003-2023)
        df = df.dropna(subset=['Year'])
        df = df[df['Year'] >= 2003].copy().reset_index(drop=True)
        df['Year'] = df['Year'].astype(int)

        print("--- Данные успешно очищены ---")
        
        # --- 3. Расчет факторов LMDI (Структура 'S' и Интенсивность 'I') ---
        
        # S_i = A_i / A (Доля сектора в экономике)
        df['S_crops'] = df['A_crops'] / df['A_total']
        df['S_livestock'] = df['A_livestock'] / df['A_total']
        
        # I_i = E_i / A_i (Выбросы на единицу продукции, "грязь" сектора)
        df['I_crops'] = df['E_crops'] / df['A_crops']
        df['I_livestock'] = df['E_livestock'] / df['A_livestock']

        # Сохраняем эту таблицу с факторами в Excel
        factors_filename = "kazakhstan_lmdi_factors.xlsx"
        df.to_excel(factors_filename, index=False, sheet_name='LMDI_Factors')
        print(f"Таблица с факторами (S и I) сохранена: {factors_filename}")
        
        # --- 4. Расчет эффектов LMDI (Год к году) ---
        
        results = []
        # Начинаем с 1, так как нам нужен t-1 (предыдущий год)
        for i in range(1, len(df)):
            t = i   # Индекс текущего года
            t0 = i-1 # Индекс предыдущего года
            
            # --- Получаем данные для t и t-1 ---
            year_t = df.loc[t, 'Year']
            
            # Активность (A)
            A_total_t = df.loc[t, 'A_total']
            A_total_t0 = df.loc[t0, 'A_total']
            
            # Секторы
            sectors = ['crops', 'livestock']
            
            # Переменные для суммирования эффектов
            delta_E_activity = 0
            delta_E_structure = 0
            delta_E_intensity = 0
            
            for sector in sectors:
                # E_i
                E_i_t = df.loc[t, f'E_{sector}']
                E_i_t0 = df.loc[t0, f'E_{sector}']
                
                # S_i
                S_i_t = df.loc[t, f'S_{sector}']
                S_i_t0 = df.loc[t0, f'S_{sector}']
                
                # I_i
                I_i_t = df.loc[t, f'I_{sector}']
                I_i_t0 = df.loc[t0, f'I_{sector}']
                
                # Рассчитываем логарифмический вес (W_i)
                W_i = lmdi_weight(E_i_t, E_i_t0)
                
                # Рассчитываем эффекты для этого сектора и добавляем к сумме
                delta_E_activity += W_i * np.log(A_total_t / A_total_t0)
                delta_E_structure += W_i * np.log(S_i_t / S_i_t0)
                delta_E_intensity += W_i * np.log(I_i_t / I_i_t0)
                
            # Общее изменение выбросов (факт)
            delta_E_total_actual = df.loc[t, 'E_total'] - df.loc[t0, 'E_total']
            
            # Сумма эффектов (расчетная)
            delta_E_total_calculated = delta_E_activity + delta_E_structure + delta_E_intensity
            
            # Сохраняем результаты
            results.append({
                'Year': int(year_t),
                'Delta_E_Total_Actual': delta_E_total_actual,
                'Effect_Activity': delta_E_activity,
                'Effect_Structure': delta_E_structure,
                'Effect_Intensity': delta_E_intensity,
                'Delta_E_Total_Calculated (Check)': delta_E_total_calculated
            })

        # --- 5. Сохранение и вывод итоговых результатов ---
        df_results = pd.DataFrame(results)
        
        # Округляем для читаемости
        df_results = df_results.round(2)
        
        results_filename = "kazakhstan_lmdi_results.xlsx"
        df_results.to_excel(results_filename, index=False, sheet_name='LMDI_Results')
        print(f"Итоговая таблица с LMDI-эффектами сохранена: {results_filename}")
        
        print("
--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ LMDI-АНАЛИЗА (2004-2023) ---")
        print(df_results.to_markdown(index=False))

    except FileNotFoundError:
        print("ОШИБКА: Файл 'date_LMDI_ARDL.csv' не найден.")
        print("Пожалуйста, убедитесь, что файл 'date_LMDI_ARDL.csv' загружен.")
    except Exception as e:
        print(f"Произошла критическая ошибка: {e}")

# --- 7. Основная точка входа ---
if __name__ == "__main__":
    run_lmdi_analysis()