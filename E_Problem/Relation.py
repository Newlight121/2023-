import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

table1 = pd.read_excel("表1-患者列表及临床信息.xlsx", index_col=0)
# table1_100 = pd.read_excel("表1-前100.xlsx", index_col=0)
table2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx", index_col=0)
table3_ED = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="ED")
table3_Hemo = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="Hemo")
table4 = pd.read_excel("表4-答案文件.xlsx")
table_ex1 = pd.read_excel("附表1-检索表格-流水号vs时间.xlsx", index_col=0)

train_index = ['sub' + str(i + 1).zfill(3) for i in range(100)]
measure = ['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']
measure_feature = table1[measure].loc[train_index]

HM_time_precess = pd.read_excel("HM time insert_get.xlsx", index_col=0).loc[train_index]
HM_value_precess = pd.read_excel("HM value insert_get.xlsx", index_col=0).loc[train_index]
ED_time_precess = pd.read_excel("ED time insert_get.xlsx", index_col=0).loc[train_index]
ED_value_precess = pd.read_excel("ED value insert_get.xlsx", index_col=0).loc[train_index]


def get_diff_value(original_table):
    # 计算原表格的行数和列数
    tmp_table = original_table.to_numpy()
    num_rows, num_cols = tmp_table.shape
    new_table = np.zeros((num_rows ** 2, num_cols))
    # 循环遍历每一行
    for i in range(num_rows):
        # 循环遍历每一行的每一行
        for j in range(num_rows):
            # 计算当前行与所有行之间的差值，并将结果存储在新表格中
            diff = tmp_table[i, :] - tmp_table[j, :]
            new_table[i * num_rows + j, :] = diff

    return new_table


HM_diff_time_precess = pd.DataFrame(get_diff_value(HM_time_precess), columns=HM_time_precess.columns)
HM_diff_value_precess = pd.DataFrame(get_diff_value(HM_value_precess), columns=HM_value_precess.columns)
ED_diff_time_precess = pd.DataFrame(get_diff_value(ED_time_precess), columns=ED_time_precess.columns)
ED_diff_value_precess = pd.DataFrame(get_diff_value(ED_value_precess), columns=ED_value_precess.columns)
measure_diff_feature = pd.DataFrame(get_diff_value(measure_feature), columns=measure_feature.columns)

select_feature = list(range(8)) + ['max', 'min', 'mean']

HM_diff_time_feature = HM_diff_time_precess[select_feature]
HM_diff_value_feature = HM_diff_value_precess[select_feature]
ED_diff_time_feature = ED_diff_time_precess[select_feature]
ED_diff_value_feature = ED_diff_value_precess[select_feature]

HM_diff_time_feature = HM_diff_time_feature.rename(columns=lambda x: "HM_time_" + str(x))
HM_diff_value_feature = HM_diff_value_feature.rename(columns=lambda x: "HM_value_" + str(x))
ED_diff_time_feature = ED_diff_time_feature.rename(columns=lambda x: "ED_time_" + str(x))
ED_diff_value_feature = ED_diff_value_feature.rename(columns=lambda x: "ED_value_" + str(x))

table_feature = pd.merge(HM_diff_time_feature, HM_diff_value_feature, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, ED_diff_time_feature, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, ED_diff_value_feature, how='left', left_index=True,
                         right_index=True)

all_feature = pd.merge(table_feature, measure_diff_feature, how='left', left_index=True,
                       right_index=True)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
all_feature_fit = sc.fit_transform(all_feature)

all_feature = pd.DataFrame(all_feature_fit, columns=all_feature.columns)

import seaborn as sns

data = all_feature
rho = data.corr(method='spearman')
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(30, 20))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(rho, cmap=cmap, annot=True, vmin=-1).get_figure().savefig('相关性分析/2-3相关性分析.png',
                                                                      dpi=800,
                                                                      bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
plt.show()

rho.to_excel('相关性分析/2-3相关性分析结果.xlsx')
