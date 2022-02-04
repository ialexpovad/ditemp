import spe
import datetime
import os
import matplotlib.pyplot as plt
import numpy
import matplotlib.ticker as ticker
from scipy.stats import norm, shapiro, normaltest, ttest_ind, t
import statistics
import pandas
# import statsmodels.formula.api as smf
import statsmodels.api as sm

def encodingfile(Path):
  from chardet.universaldetector import UniversalDetector
  '''
    Функция возвращает кодировку указанного файла в формате строки (строковый тип даннных).
    :param Path: Директроия, где располагается файл.
    :param NameFile: Имя файла с расширением (namefile.txt)
    :return: Кодировка файла ('utf-8')
    '''
  enc = UniversalDetector()
  with open(Path, 'rb') as flop:
    for line in flop:
        enc.feed(line)
        if enc.done:
            break
    enc.close()
    return enc.result['encoding']
LISTTEMP = list()
LISTTIME = list()

path = 'spectrum/04.02.2022'

for item in os.listdir(path):
    data = spe.reading(path + f'/{item}', enc = 'utf-16')
    LISTTEMP.append(data['temp'])
    LISTTIME.append(str(data['Time']).split(" ")[1][0:-3])

zipped = list(zip(LISTTIME, LISTTEMP))
dataframe = pandas.DataFrame(zipped, columns= ('Time', 'Temp'))
# print(dataframe)
savecsv = dataframe.to_csv(r'savetemp/diffcsv.csv', header = True)

# __________________ Статистика __________________ #
# https://python-school.ru/blog/python-for-statissticians/
# Первый критерий Шапиро-Уилк
stat, p = shapiro(dataframe['Temp'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности')
else:
    print('Отклонить гипотезу о нормальности')

# Второй тест по критерию согласия Пирсона
stat, p = normaltest(dataframe['Temp']) # Критерий согласия Пирсона
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Принять гипотезу о нормальности')
else:
    print('Отклонить гипотезу о нормальности')

# T-тест (или тест Стьюдента) решает задачу доказательства наличия
# различий средних значений количественной переменной в случае, когда имеются лишь две сравниваемые группы.
half = len(dataframe['Temp']) / 2
sam1 = dataframe.loc[:half, 'Temp']
sam2 = dataframe.loc[half:, 'Temp']
tt = ttest_ind(sam2, sam1)
print(tt)

dfs = (half - 1) + (half - 1)
ttppf = t.ppf(0.975, dfs)
print()
print(ttppf)
# ttp меньше tt. Это значит, можем отвергать нулеувю гипотезу.

# Линейная регрессия в Statsmodel
x = numpy.linspace(0, len(dataframe["Temp"]), len(dataframe["Temp"]))
y = dataframe['Temp'].tolist()
x = sm.add_constant(x)
result = sm.OLS(y, x).fit()
# result.save("result_{}.txt".format(format(datetime.date.today())))
# from PIL import Image, ImageDraw, ImageFont
# image = Image.new('RGB', (800, 400))
# draw = ImageDraw.Draw(image)
# font = ImageFont.truetype("arial.ttf", 16)
# draw.text((0, 0), str(result.summary()), font=font)
# image = image.convert('1') # bw
# image = image.resize((600, 300), Image.ANTIALIAS)
# image.save('output.png')
#

plt.rc('figure')
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output_{}.png'.format(datetime.date.today()), dpi = 1000)

# print(result.summary())
# __________________ Статистика __________________ #

# __________________ Графики __________________ #
fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(8, 4)
)
ax1.plot(LISTTIME, LISTTEMP, 'k-', label = 'Curve temp')
ax1.set_xlabel('Time')
ax1.set_ylabel('Temp, $^oC$')
# ax1.set_title('Зависимость изменения температуры со временем \n в кабинете 315 Date: {}'.format(datetime.date.today()))
ax1.grid(True)
ax1.legend(loc='upper left')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))

LISTTEMP.sort()
mean = statistics.mean(LISTTEMP)
cko = statistics.stdev(LISTTEMP)
pdf = norm.pdf(LISTTEMP, mean, cko)
x = numpy.linspace(15, 30, 1000)
xmean = numpy.mean(x)
xstd = numpy.std(x)
XPDF = norm.pdf(x, mean, cko)
ax2.plot(LISTTEMP, pdf,'k--', linewidth = 1)
ax2.plot(x, XPDF, 'r-', linewidth = 0.5)
# ax2.text(16, 0.35, 'Среднее = {}'.format(round(mean,3)))
# ax2.text(16, 0.33,  'СКО = {}'.format(round(cko,3)))
ax2.grid(True)
plt.savefig('temp_{}.png'.format(datetime.date.today()), dpi = 1000)
plt.show()

# print(mean, cko)
# __________________ Графики __________________ #
