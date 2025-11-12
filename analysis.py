import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import io
import sys
import numpy as np

# --- 0. Настройка окружения ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Определение констант ---
# Карта для перевода с русского на английский (для файла GVA)
country_name_map = {
    'Казахстан': 'Kazakhstan',
    'Таджикистан': 'Tajikistan',
    'Узбекистан': 'Uzbekistan',
    'Кыргызстан': 'Kyrgyzstan'
}
# Итоговый список стран на английском
countries_en = ['Kazakhstan', 'Tajikistan', 'Uzbekistan', 'Kyrgyzstan']

def get_decoupling_status(gdp_growth, em_growth):
    """
    Определяет статус декуплинга (на английском) на основе модели Tapio.
    """
    if pd.isna(gdp_growth) or pd.isna(em_growth):
        return "No Data"

    # 1. Сильный декуплинг
    if gdp_growth > 0 and em_growth < 0:
        return "Strong Decoupling"
    
    # 2. Слабый декуплинг
    if gdp_growth > 0 and em_growth > 0:
        if gdp_growth == 0: return "Not Classified"
        elasticity = em_growth / gdp_growth
        if elasticity < 0.8:
            return "Weak Decoupling"
        # 3. Связанность
        elif elasticity <= 1.2:
            return "Coupling"
        # 4. Сильная связанность
        else:
            return "Strong Coupling"

    # 5. Рецессия
    if gdp_growth < 0 and em_growth < 0:
        if gdp_growth == 0: return "Not Classified"
        elasticity = em_growth / gdp_growth
        if elasticity > 1.2:
             return "Weak Recession"
        elif elasticity >= 0.8:
            return "Coupled Recession"
        else:
            return "Strong Recession"

    # 6. Сильная негативная связанность
    if gdp_growth < 0 and em_growth > 0:
        return "Strong Negative Coupling"

    return "Not Classified"

try:
    # --- 2. Загрузка данных (с английскими именами файлов) ---
    
    # Загрузка данных по % от ВВП (это уже % роста ВДС)
    df_gva_growth = pd.read_csv(
        "data_gva_growth.csv", # Имя файла, которое вы загрузили
        header=1,
        sep=','
    )
    
    # Загрузка данных по Выбросам (абсолютные значения)
    df_emissions_abs = pd.read_csv(
        "data_emissions.csv", # Имя файла, которое вы загрузили
        header=1,
        sep=','
    )

    # --- 3. Очистка и Переименование ---
    
    # 3.1. Файл ВВП (% Роста)
    # Переименовываем первый столбец
    df_gva_growth.rename(columns={df_gva_growth.columns[0]: 'Year'}, inplace=True)
    # Переименовываем русские названия стран в английские
    df_gva_growth.rename(columns=country_name_map, inplace=True)
    df_gva_growth['Year'] = pd.to_numeric(df_gva_growth['Year'], errors='coerce')
    # Оставляем только нужные колонки
    df_gva_growth = df_gva_growth[['Year'] + countries_en]

    # 3.2. Файл Выбросов (kt)
    # В этом файле столбцы уже на английском
    df_emissions_abs['Year'] = pd.to_numeric(df_emissions_abs['Year'], errors='coerce')
    df_emissions_abs = df_emissions_abs[['Year'] + countries_en]

    # --- 4. Объединение и Расчеты ---
    # Объединяем два датафрейма по году
    df_merged = pd.merge(df_gva_growth, df_emissions_abs, on='Year', how='inner', suffixes=('_GVA_Growth', '_Emissions_Abs'))

    df_merged.sort_values(by='Year', inplace=True)
    df_merged['Year'] = df_merged['Year'].astype(int)

    print("Data successfully loaded and merged.")
    print("\n--- CALCULATING DECOUPLING COEFFICIENTS ---")
    
    all_tables_list = []

    for country in countries_en:
        # Имена столбцов
        gva_growth_col = f'{country}_GVA_Growth'
        emissions_abs_col = f'{country}_Emissions_Abs'
        
        # 1. Рост ВДС (%)
        gva_growth = pd.to_numeric(df_merged[gva_growth_col], errors='coerce')
        
        # 2. Рост Выбросов (%)
        emissions_abs = pd.to_numeric(df_merged[emissions_abs_col], errors='coerce')
        emissions_growth_col = f'{country}_Emissions_Growth_%'
        emissions_growth = emissions_abs.pct_change() * 100
        df_merged[emissions_growth_col] = emissions_growth # Сохраняем для графика
        
        # 3. Эластичность (Коэфф. декуплинга)
        elasticity = emissions_growth / gva_growth
        
        # 4. Определение статуса (на английском)
        status = [get_decoupling_status(g, e) for g, e in zip(gva_growth, emissions_growth)]
        
        # Собираем таблицу для страны
        country_table = pd.DataFrame({
            'Year': df_merged['Year'],
            'GVA Growth (%)': gva_growth.round(2),
            'Emissions Growth (%)': emissions_growth.round(2),
            'Elasticity Coeff.': elasticity.replace([np.inf, -np.inf], np.nan).round(3),
            'Status': status
        })
        
        country_table['Country'] = country # Добавляем колонку "Country"
        all_tables_list.append(country_table)

    # --- 5. Экспорт Таблиц ---
    
    # Объединяем все таблицы в одну
    final_df = pd.concat(all_tables_list)
    final_df = final_df[['Country', 'Year', 'GVA Growth (%)', 'Emissions Growth (%)', 'Elasticity Coeff.', 'Status']]
    final_df.sort_values(by=['Country', 'Year'], inplace=True)
    
    # Фильтруем, чтобы убрать годы до 2003 (т.к. ВДС начинается с 2003)
    final_df = final_df[final_df['Year'] >= 2003]
    df_merged = df_merged[df_merged['Year'] >= 2003]
    
    # Сохраняем в CSV (с разделителем ';')
    csv_filename = "decoupling_analysis_v2_en.csv"
    final_df.to_csv(csv_filename, index=False, sep=';', encoding='utf-8-sig')
    print(f"All analysis data saved to CSV: {csv_filename}")
    
    # Сохраняем в Excel
    excel_filename = "decoupling_analysis_v2_en.xlsx"
    final_df.to_excel(excel_filename, index=False, sheet_name='Decoupling_Analysis')
    print(f"All analysis data saved to Excel: {excel_filename}")
    
    # Выводим сводную таблицу (на английском)
    print("\n--- DECOUPLING ANALYSIS SUMMARY (2003-2023) ---")
    print(final_df.to_markdown(index=False))

    # --- 6. ПОСТРОЕНИЕ ГРАФИКОВ (на английском) ---
    print("\n--- PLOTTING CHARTS (% GROWTH vs % GROWTH) ---")
    plt.style.use('seaborn-v0_8-whitegrid')

    for country in countries_en:
        gva_growth_col = f'{country}_GVA_Growth'
        emissions_growth_col = f'{country}_Emissions_Growth_%'
        
        if gva_growth_col not in df_merged.columns or emissions_growth_col not in df_merged.columns:
            print(f"Skipping {country}: data not found.")
            continue

        fig, ax1 = plt.subplots(figsize=(12, 7))
        year = df_merged['Year']
        gva_growth_data = pd.to_numeric(df_merged[gva_growth_col], errors='coerce')
        emissions_growth_data = df_merged[emissions_growth_col]

        # --- Левая ось Y (Линейный график - РОСТ ВЫБРОСОВ %) ---
        color_emissions = '#D35400'
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Emissions Growth (%)', color=color_emissions, fontsize=12)
        line_plot = ax1.plot(year, emissions_growth_data, color=color_emissions, marker='o', linestyle='-', linewidth=2, label='Emissions Growth (%)')
        ax1.tick_params(axis='y', labelcolor=color_emissions)
        ax1.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x)}%'))

        # Настройка оси X
        ax1.set_xticks(year)
        ax1.set_xticklabels(year, rotation=45, ha='right')
        ax1.set_xlim(df_merged['Year'].min() - 0.5, df_merged['Year'].max() + 0.5)

        # --- Правая ось Y (Столбчатая диаграмма - РОСТ ВВП %) ---
        ax2 = ax1.twinx()
        color_gdp = '#34568B'
        ax2.set_ylabel('Agricultural GVA Growth (%)', color=color_gdp, fontsize=12)
        bar_plot = ax2.bar(year, gva_growth_data, color=color_gdp, label='Agricultural GVA Growth (%)', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_gdp)
        ax2.axhline(0, color='grey', linewidth=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x)}%'))

        # --- Заголовок и Легенда ---
        plt.title(f'Decoupling Analysis: {country}', fontsize=16, fontweight='bold')
        plots = line_plot + [bar_plot]
        labels = [p.get_label() for p in plots]
        ax1.legend(plots, labels, loc='upper left', frameon=True, shadow=True) 
        
        fig.tight_layout()
        
        filename = f"decoupling_analysis_growth_pct_en_{country}.png"
        plt.savefig(filename)
        print(f"Chart saved as: {filename}")
        plt.close(fig)

    print("\nAll charts have been plotted.")

except FileNotFoundError as e:
    print("-------------------------------------------------------------------")
    print(f"CRITICAL ERROR: FileNotFoundError. File not found: {e.filename}")
    # Сообщаем пользователю об ожидаемых английских именах
    print("Please ensure the files 'data_gva_growth.csv' and 'data_emissions.csv' are in the same directory.")
    print("-------------------------------------------------------------------")
except Exception as e:
    print(f"A critical error occurred: {e}")
    print(f"ERROR DETAILS: {e}")
    print("Script execution stopped.")

print("Code execution finished.")
