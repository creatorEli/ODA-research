import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np


#########################################################################################
#                                                                                       #
#       ВАЖНО!!!!  !!!! !!!! !!!!                                                                      #
#                                                                                       # 
#   датасет в форме (19.9МБ) урезен по сравнению с исходным (1.7 ГБ)                    #
#   ПОСКОЛЬКУ ИСХОДНИК НЕ ВЛЕЗАЛ В ФОРМУ!!!                                             #
#   НА ДИАГРАММАХ РЕЗУЛЬТАТЫ МОГУТ ОТЛИЧАТЬСЯ ИЗ-ЗА НЕРЕПРЕЗЕНАТАТИВНОСТИ!!             #
#   (В ИСХОДНОМ ДАТАСЕТЕ 1.6 МЛН СТРОК, В СОКРАЩЕННОМ ВСЕГО 18940 !!!!!                 #
#                                                                                       #
#   ИСХОДНЫЙ ДАТАСЕТ ПО ССЫЛКЕ                                                          #
#   https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data      #
#                                                                                       #
#########################################################################################

PATH_TO_CSV = "python\job_descriptions.csv"
data = pd.read_csv(PATH_TO_CSV)


def parse_salary(salary_str):
    """
    Преобразует строку формата "$<мин>K-$<макс>K" в два числа.
    Пример: "$50K-$80K" → (50000, 80000)
    """
    if pd.isna(salary_str):
        return np.nan, np.nan
    try:
        # Убираем пробелы и символы валюты
        salary_str = str(salary_str).strip().replace('$', '').replace(',', '')
        # Разделяем на минимальную и максимальную части
        if '-' in salary_str:
            min_part, max_part = salary_str.split('-')
        else:
            # Если нет диапазона, используем одно значение для обоих
            min_part = max_part = salary_str
        # Обрабатываем минимальную зарплату
        if 'K' in min_part.upper():
            min_salary = int(min_part.upper().replace('K', '')) * 1000
        elif 'M' in min_part.upper():  # На случай, если есть миллионы
            min_salary = int(min_part.upper().replace('M', '')) * 1000000
        else:
            min_salary = int(min_part)
        # Обрабатываем максимальную зарплату
        if 'K' in max_part.upper():
            max_salary = int(max_part.upper().replace('K', '')) * 1000
        elif 'M' in max_part.upper():
            max_salary = int(max_part.upper().replace('M', '')) * 1000000
        else:
            max_salary = int(max_part)
        return int(min_salary), int(max_salary)
    except Exception as e:
        print(f"Ошибка при обработке значения: {salary_str}. Ошибка: {e}")
        return np.nan, np.nan
# Применяем функцию к столбцу Salary
salary_data = data['Salary Range'].apply(parse_salary)
# Создаем новые столбцы
data['Salary_Min'] = salary_data.apply(lambda x: x[0])
data['Salary_Max'] = salary_data.apply(lambda x: x[1])
data['Salary_Avg'] = salary_data.apply(lambda x: (x[0] + x[1]) / 2 if pd.notna(x[0]) and pd.notna(x[1]) else np.nan)

def parse_experience(exp_str):
    """
    Преобразует строку формата "<мин> to <макс> Years" в два числа.
    Пример: "3 to 5 Years" → (3, 5)
    """
    if pd.isna(exp_str):
        return np.nan, np.nan
    
    try:
        exp_str = str(exp_str).strip().lower()
        # Убираем слово "years" и лишние пробелы
        exp_str = exp_str.replace('years', '').replace('year', '').strip()
        # Обрабатываем разные форматы
        if 'to' in exp_str:
            min_part, max_part = exp_str.split('to')
        # Преобразуем в числа
        min_exp = int(min_part.strip())
        max_exp = int(max_part.strip())
        return min_exp, max_exp
    except Exception as e:
        # Пробуем извлечь числа из строки
        import re
        numbers = re.findall(r'\d+', exp_str)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        elif len(numbers) == 1:
            return int(numbers[0]), int(numbers[0])
        else:
            print(f"Не удалось обработать опыт: {exp_str}. Ошибка: {e}")
            return np.nan, np.nan
# Применяем функцию к столбцу Experience
exp_data = data['Experience'].apply(parse_experience)
# Создаем новые столбцы
data['Experience_Min'] = exp_data.apply(lambda x: x[0])
data['Experience_Max'] = exp_data.apply(lambda x: x[1])
print(data.head(5))



# АНАЛИЗ 1 
print("="*60)
print("АНАЛИЗ 1")
print("="*60)

data_an1 = data.copy()
# Создаем числовой показатель для уровня образования
def education_to_numeric(qualification):
    """
    Преобразует квалификацию в числовой показатель.
    Шкала: 0 - без требований, 1 - среднее, 2 - бакалавр, 3 - магистр, 4 - PhD
    """
    if pd.isna(qualification):
        return 0
    
    qual = str(qualification).lower()
    
    if any(word in qual for word in ['phd', 'doctorate', 'doctoral']):
        return 4
    elif any(word in qual for word in ['master', 'mba', 'msc', 'mca', 'ma', 'm.tech', 'm.eng', 'm.com']):
        return 3
    elif any(word in qual for word in ['bachelor', 'bca', 'bsc', 'bba', 'ba', 'b.com', 'b.tech', 'b.eng', 'undergraduate']):
        return 2
    elif any(word in qual for word in ['high school', 'secondary', 'diploma', 'associate']):
        return 1
    else:
        #print(qual)
        return 0  # для других случаев

# Применяем функцию

data_an1.drop(['latitude', 'longitude', 'Company Size'], axis=1)
data_an1['Education_Score'] = data_an1['Qualifications'].apply(education_to_numeric)

print("ПОДГОТОВКА ДАННЫХ ДЛЯ АНАЛИЗА\n\n")
print(f"Размер выборки для анализа: {len(data_an1)} записей")
print(f"\nРаспределение уровней образования:")
print(data_an1['Education_Score'].value_counts().sort_index())
print("\nОписательная статистика:")
print(data_an1.describe())

# Создаем графики для визуального анализа
fig, axes = plt.pyplot.subplots(1, 3, figsize=(18, 6))


# 1. Столбчатая диаграмма зарплаты по опыту
exp_salary_ungrouped = data_an1.groupby('Experience_Min')['Salary_Avg'].agg(['mean', 'std', 'count']).reset_index()
exp_salary_ungrouped = exp_salary_ungrouped.sort_values('Experience_Min')

# Создаем столбчатую диаграмму
bars = axes[0].bar(exp_salary_ungrouped['Experience_Min'], 
                  exp_salary_ungrouped['mean'], 
                  width=0.6, color='skyblue', 
                  edgecolor='navy', linewidth=1.5, alpha=0.9,
                  yerr=exp_salary_ungrouped['std'], capsize=3, error_kw={'elinewidth': 1})

axes[0].set_xlabel('Требуемый опыт (лет)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Зависимость зарплаты от требуемого опыта', fontsize=14, fontweight='bold', pad=20)
axes[0].grid(True, alpha=0.3, axis='y')

min_exp = int(exp_salary_ungrouped['Experience_Min'].min())
max_exp = int(exp_salary_ungrouped['Experience_Min'].max())
axes[0].set_xticks(range(min_exp, max_exp + 1))
axes[0].set_xticklabels(range(min_exp, max_exp + 1))

# Добавляем значения над столбцами (только если есть данные)
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not np.isnan(height) and height > 0:
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${height:,.0f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')

# 2. Зарплата от уровня образования
education_labels = {0: 'Нет требований', 1: 'Среднее', 2: 'Бакалавр', 3: 'Магистр', 4: 'PhD'}
education_mapping = data_an1.copy()
education_mapping['Education_Label'] = education_mapping['Education_Score'].map(education_labels)

# Группируем по уровню образования и считаем среднюю зарплату
edu_salary = education_mapping.groupby('Education_Label')['Salary_Avg'].agg(['mean', 'std', 'count'])
edu_order = ['Нет требований', 'Среднее', 'Бакалавр', 'Магистр', 'PhD']
edu_salary = edu_salary.reindex(edu_order)

# Строим столбчатую диаграмму с тем же стилем
bars2 = axes[1].bar(range(len(edu_salary)), edu_salary['mean'], 
                   yerr=edu_salary['std'], capsize=5, color='lightgreen', 
                   edgecolor='darkgreen', linewidth=1.5, alpha=0.9)
axes[1].set_xlabel('Уровень образования', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
axes[1].set_title('Зависимость зарплаты от образования', fontsize=14, fontweight='bold', pad=20)
axes[1].set_xticks(range(len(edu_salary)))
axes[1].set_xticklabels(edu_salary.index, rotation=45, fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

# Добавляем значения над столбцами
for i, bar in enumerate(bars2):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Совместное влияние опыта и образования
scatter = axes[2].scatter(data_an1['Experience_Min'], 
                            data_an1['Salary_Avg'], 
                            c=data_an1['Education_Score'], 
                            cmap='viridis', alpha=0.7)
axes[2].set_xlabel('Средний требуемый опыт (лет)')
axes[2].set_ylabel('Средняя зарплата ($)')
axes[2].set_title('Зарплата: опыт vs образование')
plt.pyplot.colorbar(scatter, ax=axes[2], label='Уровень образования')
axes[2].grid(True, alpha=0.3)

plt.pyplot.tight_layout()
plt.pyplot.show()






################ АНАЛИЗ 2 ###############################

# ВТОРАЯ ГИПОТЕЗА: Влияние специализированных технических и управленческих навыков на зарплату
print("="*80)
print("ГИПОТЕЗА 2: Вакансии с узкоспециализированными техническими навыками предлагают более")
print("высокую зарплату, но при добавлении управленческих навыков относительная премия снижается")
print("="*80)

# Создаем копию данных для анализа
data_an2 = data.copy()

# 1. Определяем списки навыков
# Технические (узкоспециализированные) навыки
tech_skills_keywords = [
    # Программирование
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'scala', 'kotlin', 'programming',
    # Веб-разработка
    'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'laravel', 'backend', 'web', 'html'
    # Базы данных
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'oracle',
    # DevOps и инфраструктура
    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform', 'ansible', 'jenkins',
    # Data Science и ML
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data science', 'big data', 'data analysis',
    # Специализированные технологии
    'blockchain', 'cybersecurity', 'embedded', 'iot', 'ar/vr', 'computer vision',
    # Другие технические
    'devops', 'sre', 'qa', 'testing', 'automation', 'cad', 'protocols', 'firewall', 'autocad', 'engineering',
    'data', 'troubleshooting', 'database', 'hardware', 'software', 'operating system'
]

# Управленческие навыки
management_skills_keywords = [
    'management', 'leadership', 'team lead', 'project management', 'agile', 'scrum',
    'stakeholder', 'strategic', 'budget', 'planning', 'mentoring', 'coaching',
    'people management', 'resource management', 'risk management', 'product management',
    'program management', 'portfolio management', 'kanban'
]

# 2. Функция для поиска навыков в тексте
def find_skills(text, skill_list):
    """Находит навыки из списка в тексте"""
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    found_skills = []
    
    for skill in skill_list:
        if skill in text:
            found_skills.append(skill)
    
    return len(found_skills)

# 3. Создаем новые колонки
# Находим навыки в столбцах Skills и Responsibilities
data_an2['Tech_Skills_Count'] = data_an2['skills'].apply(
    lambda x: find_skills(x, tech_skills_keywords)
)
data_an2['Mgmt_Skills_Count'] = data_an2['skills'].apply(
    lambda x: find_skills(x, management_skills_keywords)
)

# Также проверяем Responsibilities
data_an2['Tech_in_Resp'] = data_an2['Responsibilities'].apply(
    lambda x: find_skills(x, tech_skills_keywords)
)
data_an2['Mgmt_in_Resp'] = data_an2['Responsibilities'].apply(
    lambda x: find_skills(x, management_skills_keywords)
)

# Суммируем навыки из обоих источников
data_an2['Total_Tech_Skills'] = data_an2['Tech_Skills_Count'] + data_an2['Tech_in_Resp']
data_an2['Total_Mgmt_Skills'] = data_an2['Mgmt_Skills_Count'] + data_an2['Mgmt_in_Resp']

# 4. Классифицируем вакансии по типам навыков
def classify_skills(row):
    tech = row['Total_Tech_Skills']
    mgmt = row['Total_Mgmt_Skills']
    
    if tech > 2 and mgmt == 0:
        return 'High_Tech_Only'
    elif tech > 2 and mgmt >= 1:
        return 'High_Tech_with_Mgmt'
    elif 1 <= tech <= 2 and mgmt == 0:
        return 'Mid_Tech_Only'
    elif 1 <= tech <= 2 and mgmt >= 1:
        return 'Mid_Tech_with_Mgmt'
    elif tech == 0 and mgmt >= 1:
        return 'Mgmt_Only'
    else:
        return 'Other'

data_an2['Skill_Category'] = data_an2.apply(classify_skills, axis=1)

# 5. Анализ зарплат по категориям навыков
print("\nРАСПРЕДЕЛЕНИЕ ВАКАНСИЙ ПО ТИПАМ НАВЫКОВ:")
category_counts = data_an2['Skill_Category'].value_counts()
for category, count in category_counts.items():
    percentage = (count / len(data_an2)) * 100
    print(f"  {category}: {count} вакансий ({percentage:.1f}%)")

print("\nСРЕДНИЕ ЗАРПЛАТЫ ПО КАТЕГОРИЯМ НАВЫКОВ:")
salary_by_category = data_an2.groupby('Skill_Category')['Salary_Avg'].agg([
    'mean', 'median', 'std', 'count'
]).round(0)

print(salary_by_category.sort_values('mean', ascending=False))

# 6. Статистический анализ
print("\nСТАТИСТИЧЕСКИЙ АНАЛИЗ:")

# Сравниваем High_Tech_Only и High_Tech_with_Mgmt
group_tech_only = data_an2[data_an2['Skill_Category'] == 'High_Tech_Only']['Salary_Avg']
group_tech_mgmt = data_an2[data_an2['Skill_Category'] == 'High_Tech_with_Mgmt']['Salary_Avg']

if len(group_tech_only) > 10 and len(group_tech_mgmt) > 10:
    
    print(f"\nСравнение High_Tech_Only vs High_Tech_with_Mgmt:")
    print(f"Средняя зарплата (только тех): ${group_tech_only.mean():,.0f}")
    print(f"Средняя зарплата (тех + менеджмент): ${group_tech_mgmt.mean():,.0f}")
    print(f"Разница: ${group_tech_mgmt.mean() - group_tech_only.mean():,.0f}")
    
    # Расчет относительной премии
    base_salary = data_an2[data_an2['Skill_Category'] == 'Other']['Salary_Avg'].mean()
    if not pd.isna(base_salary):
        premium_tech_only = (group_tech_only.mean() - base_salary) / base_salary * 100
        premium_tech_mgmt = (group_tech_mgmt.mean() - base_salary) / base_salary * 100
        
        print(f"\nОТНОСИТЕЛЬНАЯ ПРЕМИЯ (относительно 'Other' категории):")
        print(f"Базовая зарплата (Other): ${base_salary:,.0f}")
        print(f"Премия за высокие технавыки: +{premium_tech_only:.1f}%")
        print(f"Премия за технавыки + менеджмент: +{premium_tech_mgmt:.1f}%")
        print(f"Разница в премиях: {premium_tech_only - premium_tech_mgmt:.1f}%")


# 7. Визуализация
fig, axes = plt.pyplot.subplots(1, 2, figsize=(14, 8))

# График 1: Распределение зарплат по категориям навыков
categories = salary_by_category.sort_values('mean', ascending=False).index
means = salary_by_category.loc[categories, 'mean']
errors = salary_by_category.loc[categories, 'std'] / np.sqrt(salary_by_category.loc[categories, 'count'])

bars = axes[0].bar(range(len(categories)), means, yerr=errors, capsize=5,
                  color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6BAA75', '#8B8B8B'])
axes[0].set_xlabel('Категория навыков', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Зарплата по типам навыков', fontsize=14, fontweight='bold', pad=20)
axes[0].set_xticks(range(len(categories)))
axes[0].set_xticklabels(categories, rotation=25, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

# Добавляем значения
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'${height:,.0f}', ha='center', va='bottom', fontsize=9)


# График 3: Относительная премия
if 'base_salary' in locals() and not pd.isna(base_salary):
    categories_for_premium = ['High_Tech_Only', 'High_Tech_with_Mgmt', 'Mid_Tech_Only',
                             'Mid_Tech_with_Mgmt', 'Mgmt_Only']
    premiums = []
    
    for cat in categories_for_premium:
        if cat in salary_by_category.index:
            cat_salary = salary_by_category.loc[cat, 'mean']
            premium = (cat_salary - base_salary) / base_salary * 100
            premiums.append(premium)
        else:
            premiums.append(0)
    
    axes[1].bar(range(len(categories_for_premium)), premiums, color='#2E86AB')
    axes[1].set_xlabel('Категория навыков', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Относительная премия (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Премия к зарплате за навыки', fontsize=14, fontweight='bold', pad=20)
    axes[1].set_xticks(range(len(categories_for_premium)))
    axes[1].set_xticklabels(categories_for_premium, rotation=25, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Добавляем линию нуля
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Добавляем значения
    for i, premium in enumerate(premiums):
        axes[1].text(i, premium + (1 if premium >= 0 else -1),
                    f'{premium:+.1f}%', ha='center', va='bottom' if premium >= 0 else 'top',
                    fontsize=9, fontweight='bold')

plt.pyplot.tight_layout()
plt.pyplot.show()



# ГИПОТЕЗА: В больших компаниях на аналогичных позициях платят в среднем больше, чем в маленьких
print("="*80)
print("ГИПОТЕЗА: В больших компаниях на аналогичных позициях (Role) платят")
print("в среднем больше, чем в маленьких компаниях")
print("="*80)

# Создаем копию данных для анализа
data_an3 = data.copy()

# 1. Определяем категории компаний (большие vs маленькие)
print("КЛАССИФИКАЦИЯ КОМПАНИЙ ПО РАЗМЕРУ:")

print(f"Минимальное число сотрудников в компании: {data_an3['Company Size'].min()}")
print(f"Минимальное число сотрудников в компании: {data_an3['Company Size'].max()}")

if 'Company Size' in data_an3.columns:
    def classify_company_size(size):
        if size <= 20000: return "Small" 
        if size <= 50000: return "Medium" 
        if size > 50000: return "Large" 

    data_an3['Company_Category'] = data_an3['Company Size'].apply(classify_company_size)

# Статистика по категориям
category_counts = data_an3['Company_Category'].value_counts()
print("\nРаспределение компаний по категориям:")
for category, count in category_counts.items():
    percentage = (count / len(data_an3)) * 100
    print(f"  {category}: {count} компаний ({percentage:.1f}%)")




# 2. Анализ зарплат по категориям компаний в целом
print("\nОБЩИЙ АНАЛИЗ ЗАРПЛАТ ПО РАЗМЕРУ КОМПАНИЙ:")

if 'Company_Category' in data_an3.columns:
    # Общая статистика зарплат по категориям компаний
    overall_stats = data_an3.groupby('Company_Category')['Salary_Avg'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    
    print(overall_stats.sort_values('mean', ascending=False).to_string())
    
    # Проверяем гипотезу в целом
    if 'Large' in overall_stats.index and 'Small' in overall_stats.index:
        large_mean = overall_stats.loc['Large', 'mean']
        small_mean = overall_stats.loc['Small', 'mean']
        diff = large_mean - small_mean
        
        print(f"\nСРАВНЕНИЕ В ЦЕЛОМ:")
        print(f"Средняя зарплата в больших компаниях: ${large_mean:,.0f}")
        print(f"Средняя зарплата в маленьких компаниях: ${small_mean:,.0f}")
        print(f"Разница: ${diff:,.0f}")
        
        if diff > 0:
            print(f"В целом большие компании платят больше")
        else:
            print(f"В целом маленькие компании платят больше")

# 3. Анализ по конкретным позициям (Role)
print("\n" + "="*80)
print("АНАЛИЗ ПО КОНКРЕТНЫМ ПОЗИЦИЯМ (ROLE)")
print("="*80)

if 'Role' in data_an3.columns and 'Company_Category' in data_an3.columns:
    # Находим самые частые позиции (топ-20)
    top_roles = data_an3['Role'].value_counts().head(20)
    
    print(f"Анализируем {len(top_roles)} самых частых позиций:")
    
    # Создаем список для хранения результатов
    role_comparison_results = []
    
    # Анализируем каждую позицию
    for role_name, role_count in top_roles.items():
        # Фильтруем данные по позиции
        role_data = data_an3[data_an3['Role'] == role_name]
        
        # Проверяем, есть ли данные для больших и маленьких компаний
        if 'Large' in role_data['Company_Category'].values and 'Small' in role_data['Company_Category'].values:
            # Группируем по категории компании
            role_stats = role_data.groupby('Company_Category')['Salary_Avg'].agg(['mean', 'count']).round(0)
            
            # Проверяем, что есть достаточно данных
            if role_stats.loc['Large', 'count'] >= 5 and role_stats.loc['Small', 'count'] >= 5:
                large_salary = role_stats.loc['Large', 'mean']
                small_salary = role_stats.loc['Small', 'mean']
                
                # Вычисляем разницу
                if small_salary > 0:  # Чтобы избежать деления на ноль
                    diff = large_salary - small_salary
                    diff_percentage = (diff / small_salary) * 100
                    
                    # Добавляем в результаты
                    role_comparison_results.append({
                        'Role': role_name,
                        'Large_Companies': large_salary,
                        'Small_Companies': small_salary,
                        'Difference_Amount': diff,
                        'Difference_Percentage': diff_percentage,
                        'Large_Count': role_stats.loc['Large', 'count'],
                        'Small_Count': role_stats.loc['Small', 'count']
                    })
    

    comparison_df = pd.DataFrame(role_comparison_results)
        
    # Сортируем по абсолютной разнице
    comparison_df['Abs_Difference'] = comparison_df['Difference_Amount'].abs()
    comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)
    
    print(f"\nРЕЗУЛЬТАТЫ СРАВНЕНИЯ ПО ПОЗИЦИЯМ:")
    print(f"Проанализировано {len(comparison_df)} позиций с достаточным количеством данных")
    
    # Статистика по гипотезе
    positive_diff = comparison_df[comparison_df['Difference_Amount'] > 0]
    negative_diff = comparison_df[comparison_df['Difference_Amount'] < 0]
        
    print(f"\nСТАТИСТИКА ГИПОТЕЗЫ:")
    print(f"Позиций, где большие компании платят больше: {len(positive_diff)} ({len(positive_diff)/len(comparison_df)*100:.1f}%)")
    print(f"Позиций, где маленькие компании платят больше: {len(negative_diff)} ({len(negative_diff)/len(comparison_df)*100:.1f}%)")
    print(f"Позиций с одинаковой зарплатой: {len(comparison_df) - len(positive_diff) - len(negative_diff)}")
        
     # Средняя разница по всем позициям
    avg_diff_percentage = comparison_df['Difference_Percentage'].mean()
    median_diff_amount = comparison_df['Difference_Amount'].median()
    
    print(f"\nСРЕДНИЕ ПОКАЗАТЕЛИ:")
    print(f"Средняя разница в процентах: {avg_diff_percentage:+.1f}%")
    print(f"Медианная разница в деньгах: ${median_diff_amount:,.0f}")
    
    # Проверка гипотезы
    if len(positive_diff) > len(negative_diff):
        print(f"\nГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ:")
        print(f"   В {len(positive_diff)} из {len(comparison_df)} позиций большие компании платят больше")
        print(f"   Средняя премия в больших компаниях: {avg_diff_percentage:+.1f}%")
    else:
        print(f"\nГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ:")
        print(f"   В {len(negative_diff)} из {len(comparison_df)} позиций маленькие компании платят больше")
        print(f"   В среднем маленькие компании платят на {abs(avg_diff_percentage):.1f}% больше")
    
    # Показываем топ-10 позиций с самой большой разницей
    print(f"\nТОП-10 ПОЗИЦИЙ С НАИБОЛЬШЕЙ РАЗНИЦЕЙ В ЗАРПЛАТЕ:")
    top_diffs = comparison_df.head(10).copy()
    
    # Форматируем вывод
    for i, row in top_diffs.iterrows():
        if row['Difference_Amount'] > 0:
            arrow = "↑"  # Большие компании платят больше
            diff_text = f"+${row['Difference_Amount']:,.0f} ({row['Difference_Percentage']:+.1f}%)"
        else:
            arrow = "↓"  # Маленькие компании платят больше
            diff_text = f"-${abs(row['Difference_Amount']):,.0f} ({row['Difference_Percentage']:.1f}%)"
        
        print(f"{arrow} {row['Role'][:40]:40} | "
              f"Большие: ${row['Large_Companies']:,.0f} | "
              f"Маленькие: ${row['Small_Companies']:,.0f} | "
              f"Разница: {diff_text}")
    
# 4. Визуализация результатов
    print("\nВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    fig, axes = plt.pyplot.subplots(1, 2, figsize=(16, 8))
    
    # График 1: Распределение зарплат по категориям компаний
    categories = ['Small', 'Medium', 'Large']
    category_means = []
    category_stds = []
    
    for cat in categories:
        if cat in data_an3['Company_Category'].unique():
            cat_data = data_an3[data_an3['Company_Category'] == cat]['Salary_Avg']
            category_means.append(cat_data.mean())
            category_stds.append(cat_data.std())
        else:
            category_means.append(0)
            category_stds.append(0)
    
    bars1 = axes[0].bar(range(len(categories)), category_means, 
                          yerr=category_stds, capsize=5, 
                          color=['lightcoral', 'lightgreen', 'skyblue'])
    axes[0].set_xlabel('Размер компании', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
    axes[0].set_title('Общая средняя зарплата по размеру компании', 
                        fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                      f'${height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # График 2: Топ-10 позиций с наибольшей зарплатой в больших компаниях
    top_10_large = comparison_df.sort_values('Large_Companies', ascending=False).head(10)
    
    y_pos = range(len(top_10_large))
    axes[1].barh(y_pos, top_10_large['Large_Companies'], color='skyblue', alpha=0.7, label='Большие компании')
    axes[1].barh(y_pos, top_10_large['Small_Companies'], color='lightcoral', alpha=0.7, label='Маленькие компании')
        
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([role[:25] + '...' if len(role) > 25 else role 
                                   for role in top_10_large['Role']])
    axes[1].set_xlabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
    axes[1].set_title('Топ-10 позиций с самой высокой зарплатой в больших компаниях', 
                            fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].set_xlim(81000,84000)
    axes[1].grid(True, alpha=0.3, axis='x')
        
        
    plt.pyplot.tight_layout()
    plt.pyplot.show()

else:
    print("Для анализа необходимы колонки 'Role' и 'Company Size'")




# ГИПОТЕЗА 4: Сезонность зарплатных предложений
print("="*80)
print("ГИПОТЕЗА 4: Зарплатные предложения имеют сезонную зависимость от даты")
print("размещения вакансии, при этом пик зарплат приходится на начало года")
print("="*80)

# Создаем копию данных для анализа
data_an4 = data.copy()

# 1. Подготовка данных с датами
print("ПОДГОТОВКА ДАННЫХ С ДАТАМИ:")

# Проверяем наличие колонки с датой
date_column = None
for col in data_an4.columns:
    if 'date' in col.lower() or 'posting' in col.lower():
        date_column = col
        break

if date_column:
    print(f"Найдена колонка с датой: '{date_column}'")
    
    # Преобразуем в datetime
    try:
        data_an4['Job_Date'] = pd.to_datetime(data_an4[date_column], errors='coerce')
        
        # Извлекаем компоненты даты
        data_an4['Year'] = data_an4['Job_Date'].dt.year
        data_an4['Month'] = data_an4['Job_Date'].dt.month
        data_an4['Quarter'] = data_an4['Job_Date'].dt.quarter
        data_an4['Week'] = data_an4['Job_Date'].dt.isocalendar().week
        data_an4['Day'] = data_an4['Job_Date'].dt.day
        data_an4['DayOfWeek'] = data_an4['Job_Date'].dt.dayofweek  # 0 = понедельник
        data_an4['Is_StartOfYear'] = data_an4['Month'].isin([1, 2])  # Январь-февраль
        data_an4['Is_EndOfYear'] = data_an4['Month'].isin([11, 12])  # Ноябрь-декабрь
        
        # Проверяем диапазон дат
        min_date = data_an4['Job_Date'].min()
        max_date = data_an4['Job_Date'].max()
        print(f"Диапазон дат: с {min_date.date()} по {max_date.date()}")
        print(f"Всего лет: {data_an4['Year'].nunique()}")
        print(f"никальные годы: {sorted(data_an4['Year'].dropna().unique())}")
        
    except Exception as e:
        print(f"Ошибка при обработке даты: {e}")
        date_column = None
else:
    print("Колонка с датой не найдена")
    # Проверим доступные колонки
    print("  Доступные колонки:", [col for col in data_an4.columns if 'date' in col.lower() or 'time' in col.lower()])


# 2. Анализ распределения вакансий по времени
print("\nРАСПРЕДЕЛЕНИЕ ВАКАНСИЙ ПО ВРЕМЕНИ:")

if 'Year' in data_an4.columns:
    # Распределение по годам
    year_counts = data_an4['Year'].value_counts().sort_index()
    print(f"  • Распределение по годам:")
    for year, count in year_counts.items():
        if not pd.isna(year):
            print(f"      {int(year)}: {count} вакансий ({count/len(data_an4)*100:.1f}%)")

if 'Month' in data_an4.columns:
    # Распределение по месяцам
    month_counts = data_an4['Month'].value_counts().sort_index()
    print(f"  • Распределение по месяцам:")
    
    month_names = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                   'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    
    for month in range(1, 13):
        count = month_counts.get(month, 0)
        if count > 0:
            print(f"      {month_names[month-1]}: {count} вакансий ({count/len(data_an4)*100:.1f}%)")

if 'Quarter' in data_an4.columns:
    # Распределение по кварталам
    quarter_counts = data_an4['Quarter'].value_counts().sort_index()
    print(f"  • Распределение по кварталам:")
    for q in range(1, 5):
        count = quarter_counts.get(q, 0)
        if count > 0:
            print(f"      Q{q}: {count} вакансий ({count/len(data_an4)*100:.1f}%)")

# 3. Анализ зарплат по месяцам
print("\nАНАЛИЗ ЗАРПЛАТ ПО МЕСЯЦАМ:")

if 'Month' in data_an4.columns and 'Salary_Avg' in data_an4.columns:
    # Группируем по месяцам
    month_stats = data_an4.groupby('Month')['Salary_Avg'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    
    # Сортируем по среднему значению
    month_stats_sorted = month_stats.sort_values('mean', ascending=False)
    
    print("Средние зарплаты по месяцам (от самого высокого к самому низкому):")
    for month_num in month_stats_sorted.index:
        month_name = month_names[month_num-1]
        mean_salary = month_stats_sorted.loc[month_num, 'mean']
        count = month_stats_sorted.loc[month_num, 'count']
        print(f"  {month_name:10} (месяц {month_num:2d}): ${mean_salary:,.0f} (n={count})")
    
    # Определяем пиковый месяц
    peak_month_num = month_stats_sorted.index[0]
    peak_month_name = month_names[peak_month_num-1]
    peak_salary = month_stats_sorted.iloc[0]['mean']
    
    # Определяем самый низкий месяц
    lowest_month_num = month_stats_sorted.index[-1]
    lowest_month_name = month_names[lowest_month_num-1]
    lowest_salary = month_stats_sorted.iloc[-1]['mean']
    
    # Разница между пиком и минимумом
    salary_diff = peak_salary - lowest_salary
    salary_diff_percent = (salary_diff / lowest_salary) * 100
    
    print(f"\nКЛЮЧЕВЫЕ НАХОДКИ:")
    print(f"Пик зарплат: {peak_month_name} (${peak_salary:,.0f})")
    print(f"Минимум зарплат: {lowest_month_name} (${lowest_salary:,.0f})")
    print(f"Разница: ${salary_diff:,.0f} ({salary_diff_percent:+.1f}%)")
    
    # Проверяем, приходится ли пик на начало года (январь-февраль)
    is_peak_at_start = peak_month_num in [1, 2]
    
    if is_peak_at_start:
        print(f"Пик зарплат действительно приходится на начало года ({peak_month_name})")
    else:
        print(f"Пик зарплат НЕ приходится на начало года ({peak_month_name})")
    
    # Проверяем гипотезу о начале года
    start_year_months = [1, 2]
    end_year_months = [11, 12]
    
    start_year_data = data_an4[data_an4['Month'].isin(start_year_months)]['Salary_Avg']
    end_year_data = data_an4[data_an4['Month'].isin(end_year_months)]['Salary_Avg']
    other_months_data = data_an4[~data_an4['Month'].isin(start_year_months + end_year_months)]['Salary_Avg']
    
    if len(start_year_data) > 0 and len(end_year_data) > 0:
        start_mean = start_year_data.mean()
        end_mean = end_year_data.mean()
        
        print(f"\nСРАВНЕНИЕ НАЧАЛА И КОНЦА ГОДА:")
        print(f"Начало года (январь-февраль): ${start_mean:,.0f} (n={len(start_year_data)})")
        print(f"Конец года (ноябрь-декабрь): ${end_mean:,.0f} (n={len(end_year_data)})")
        print(f"Остальные месяцы: ${other_months_data.mean():,.0f} (n={len(other_months_data)})")
        
        diff_start_end = start_mean - end_mean
        if diff_start_end > 0:
            print(f"Разница начало-конец: ${diff_start_end:,.0f} (+{(diff_start_end/end_mean*100):.1f}%)")
        else:
            print(f"Разница начало-конец: ${diff_start_end:,.0f} ({(diff_start_end/end_mean*100):.1f}%)")


# 4. Визуализация результатов
print("\nВИЗУАЛИЗАЦИЯ СЕЗОННОСТИ ЗАРПЛАТ")

fig, axes = plt.pyplot.subplots(1, 1, figsize=(16, 9))

# График 1: Зарплаты по месяцам (линейный график)
if 'Month' in data_an4.columns:
    # Группируем по месяцам
    monthly_data = data_an4.groupby('Month')['Salary_Avg'].agg(['mean', 'std', 'count']).sort_index()
    
    # Создаем данные для графика
    months = monthly_data.index
    month_labels = [month_names[m-1] for m in months]
    mean_salaries = monthly_data['mean']
    std_errors = monthly_data['std'] / np.sqrt(monthly_data['count'])
    
    axes.plot(range(len(months)), mean_salaries, 
                   marker='o', linewidth=2, markersize=8, color='steelblue')
    axes.fill_between(range(len(months)), 
                           mean_salaries - std_errors, 
                           mean_salaries + std_errors, 
                           alpha=0.2, color='steelblue')
    
    # Добавляем линию тренда
    z = np.polyfit(range(len(months)), mean_salaries, 1)
    p = np.poly1d(z)
    axes.plot(range(len(months)), p(range(len(months))), 
                   "r--", alpha=0.7, label=f'Тренд: y={z[0]:.0f}x+{z[1]:.0f}')
    
    axes.set_xlabel('Месяц', fontsize=12, fontweight='bold')
    axes.set_ylabel('Средняя зарплата ($)', fontsize=12, fontweight='bold')
    axes.set_title('Сезонность зарплат по месяцам', fontsize=14, fontweight='bold')
    axes.set_xticks(range(len(months)))
    axes.set_xticklabels(month_labels, rotation=45, ha='right')
    axes.grid(True, alpha=0.3)
    axes.legend()
    
    # Выделяем начало года (январь-февраль)
    for i, month in enumerate(months):
        if month in [1, 2]:
            axes.axvspan(i-0.4, i+0.4, alpha=0.2, color='green', label='Начало года' if i==0 else "")
    
    # Добавляем аннотации для пиковых месяцев
    for i, (month, salary) in enumerate(zip(months, mean_salaries)):
        if salary == max(mean_salaries) or salary == min(mean_salaries):
            axes.annotate(f'${salary:,.0f}', 
                              xy=(i, salary), 
                              xytext=(i, salary + (salary*0.05 if salary == max(mean_salaries) else -salary*0.05)),
                              arrowprops=dict(arrowstyle="->", color='red', alpha=0.7),
                              fontsize=10, fontweight='bold', ha='center')

plt.pyplot.tight_layout()
plt.pyplot.show()



# КОРРЕЛЯЦИОННАЯ МАТРИЦА ДЛЯ ДАТАСЕТА
print("="*80)
print("КОРРЕЛЯЦИОННАЯ МАТРИЦА ДАТАСЕТА")
print("="*80)

# Создаем копию данных для анализа
data_corr = data.copy()

# 1. Выбираем только числовые колонки для корреляционного анализа
print("ВЫБОР ЧИСЛОВЫХ ПЕРЕМЕННЫХ ДЛЯ АНАЛИЗА:")

# Автоматически находим числовые колонки
numeric_cols = data_corr.select_dtypes(include=[np.number]).columns.tolist()

# Исключаем колонки с датами или индексами, если они есть
exclude_keywords = ['job id', 'latitude', 'longitude', 'date']
numeric_cols = [col for col in numeric_cols 
                if not any(keyword in col.lower() for keyword in exclude_keywords)]

print(f"  • Найдено {len(numeric_cols)} числовых колонок:")
for i, col in enumerate(numeric_cols[:20], 1):  # Показываем первые 20
    print(f"    {i:2d}. {col}")

if len(numeric_cols) > 20:
    print(f"    ... и еще {len(numeric_cols) - 20} колонок")

# 2. Строим корреляционную матрицу
print(f"\nВЫЧИСЛЕНИЕ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ...")

# Выбираем только числовые колонки для корреляции
corr_data = data_corr[numeric_cols]

# Вычисляем корреляционную матрицу
correlation_matrix = corr_data.corr()

print(f"  • Размер матрицы: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
print(f"  • Метод корреляции: Пирсона (линейная корреляция)")

# 3. Визуализация корреляционной матрицы
print(f"\nВИЗУАЛИЗАЦИЯ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ...")

# Создаем фигуру с двумя графиками
fig, ax1 = plt.pyplot.subplots(1, 1, figsize=(20, 10))

# График 1: Полная тепловая карта корреляций
mask_upper = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Используем diverging палитру для лучшего восприятия
sns.heatmap(correlation_matrix, 
            mask=mask_upper,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True, 
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
            ax=ax1)

ax1.set_title('Полная корреляционная матрица\n(треугольная маска для избежания дублирования)', 
             fontsize=14, fontweight='bold', pad=20)

# Поворачиваем метки осей для лучшей читаемости
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)


plt.pyplot.tight_layout()
plt.pyplot.show()

print("Завершение программы...")