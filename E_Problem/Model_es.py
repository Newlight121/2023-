import seaborn as sns
import pandas as pd
import numpy as np
import xgboost
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from matplotlib import pyplot as plt
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor, XGBClassifier
import warnings
import catboost as ctb
import lightgbm as LGB
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
import re

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings(action='ignore')
table1 = pd.read_excel("表1-患者列表及临床信息.xlsx", index_col=0)
# table1_100 = pd.read_excel("表1-前100.xlsx", index_col=0)
table2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx", index_col=0)
table3_ED = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="ED")
table3_Hemo = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="Hemo")
table4 = pd.read_excel("表4-答案文件.xlsx")
table_ex1 = pd.read_excel("附表1-检索表格-流水号vs时间.xlsx", index_col=0)

table1_feature_name = ['年龄', '性别', '脑出血前mRS评分', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史',
                       '吸烟史', '饮酒史', '发病到首次影像检查时间间隔', '高血压', '低血压', '脑室引流', '止血治疗',
                       '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']
table2_feature_name = ['HM_volume', 'HM_ACA_R_Ratio', 'HM_MCA_R_Ratio', 'HM_PCA_R_Ratio', 'HM_Pons_Medulla_R_Ratio',
                       'HM_Cerebellum_R_Ratio', 'HM_ACA_L_Ratio', 'HM_MCA_L_Ratio', 'HM_PCA_L_Ratio',
                       'HM_Pons_Medulla_L_Ratio', 'HM_Cerebellum_L_Ratio', 'ED_volume', 'ED_ACA_R_Ratio',
                       'ED_MCA_R_Ratio',
                       'ED_PCA_R_Ratio', 'ED_Pons_Medulla_R_Ratio', 'ED_Cerebellum_R_Ratio', 'ED_ACA_L_Ratio',
                       'ED_MCA_L_Ratio', 'ED_PCA_L_Ratio', 'ED_Pons_Medulla_L_Ratio', 'ED_Cerebellum_L_Ratio']

table3_feature_name = ['original_shape_Elongation',
                       'original_shape_Flatness',
                       'original_shape_LeastAxisLength',
                       'original_shape_MajorAxisLength',
                       'original_shape_Maximum2DDiameterColumn',
                       'original_shape_Maximum2DDiameterRow',
                       'original_shape_Maximum2DDiameterSlice',
                       'original_shape_Maximum3DDiameter',
                       'original_shape_MeshVolume',
                       'original_shape_MinorAxisLength',
                       'original_shape_Sphericity',
                       'original_shape_SurfaceArea',
                       'original_shape_SurfaceVolumeRatio',
                       'original_shape_VoxelVolume',
                       'NCCT_original_firstorder_10Percentile',
                       'NCCT_original_firstorder_90Percentile',
                       'NCCT_original_firstorder_Energy',
                       'NCCT_original_firstorder_Entropy',
                       'NCCT_original_firstorder_InterquartileRange',
                       'NCCT_original_firstorder_Kurtosis',
                       'NCCT_original_firstorder_Maximum',
                       'NCCT_original_firstorder_MeanAbsoluteDeviation',
                       'NCCT_original_firstorder_Mean',
                       'NCCT_original_firstorder_Median',
                       'NCCT_original_firstorder_Minimum',
                       'NCCT_original_firstorder_Range',
                       'NCCT_original_firstorder_RobustMeanAbsoluteDeviation',
                       'NCCT_original_firstorder_RootMeanSquared',
                       'NCCT_original_firstorder_Skewness',
                       'NCCT_original_firstorder_Uniformity',
                       'NCCT_original_firstorder_Variance'
                       ]

mRS_name = ["90天mRS"]
## 标签处理
for index, line in (table1["性别"]).items():
    if line == "男":
        table1["性别"].loc[index] = 1;
    else:
        table1["性别"].loc[index] = 0;

table1["高血压"] = ""
table1["低血压"] = ""

for index, line in (table1["血压"]).items():
    high = float(line.split('/')[0])
    low = float(line.split('/')[1])
    table1["高血压"].loc[index] = high
    table1["低血压"].loc[index] = low

## 提取table3所有记录
table3_HM_extract = {}
table3_ED_extract = {}
for index, line in (table1["入院首次影像检查流水号"]).items():
    tmp1, tmp2 = 0, 0  # 记录提取次数
    for index_tmp, table3_num in (table3_Hemo["流水号"]).items():
        if index == "sub132":
            pass
        if table3_num == line:
            table3_HM_extract[index] = table3_Hemo[table3_feature_name].loc[index_tmp]  ## 记录
            tmp1 = tmp1 + 1
            if tmp1 > 1:
                print("发生多次提取，流水号重复", table3_num, index)
    for index_tmp, table3_num in (table3_ED["流水号"]).items():
        if index == "sub132":
            pass
        if table3_num == line:
            table3_ED_extract[index] = table3_ED[table3_feature_name].loc[index_tmp]  ## 记录
            tmp2 = tmp2 + 1
            if tmp2 > 1:
                print("发生多次提取，流水号重复", table3_num, index)

df_table3_hm_extract = pd.DataFrame.from_dict(table3_HM_extract, orient='index', columns=table3_feature_name)
df_table3_ed_extract = pd.DataFrame.from_dict(table3_ED_extract, orient='index', columns=table3_feature_name)
df_table3_hm_extract = df_table3_hm_extract.rename(columns=lambda x: "HM_" + x)
df_table3_ed_extract = df_table3_ed_extract.rename(columns=lambda x: "ED_" + x)
train_index = ['sub' + str(i + 1).zfill(3) for i in range(100)]
predict_index_tmp = ['sub' + str(i + 101).zfill(3) for i in range(60)]
# predict_index_tmp.remove('sub131')
# predict_index_tmp.remove('sub132')

predict_index = train_index + predict_index_tmp

table1_feature = table1[table1_feature_name]
table2_feature = table2[table2_feature_name]

table_feature = pd.merge(table1_feature, table2_feature, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, df_table3_hm_extract, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, df_table3_ed_extract, how='left', left_index=True,
                         right_index=True)

pd_train_feature = table_feature.loc[train_index]

# f = open("特征名称.txt", "w")
# f.writelines(str(list(pd_train_feature.columns)))
# f.close()

pd_train_label = table1[mRS_name].loc[train_index]
pd_predict_feature = table_feature.loc[predict_index]

data = pd.merge(pd_train_feature, pd_train_label, how='left', left_index=True,
                right_index=True)
rho = data.corr(method='spearman')
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rho.to_excel("全部变量相关性分析.xlsx")
# plt.figure(figsize=(80, 60))
# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# sns.heatmap(rho, cmap=cmap, annot=True, vmin=-1).get_figure().savefig('相关性分析\全部变量相关性分析.png',
#                                                                       dpi=500,
#                                                                       bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
# plt.show()

'''
select_feature = ['低血压', '高血压', '年龄', '发病到首次影像检查时间间隔', 'HM_original_shape_Elongation',
                  'HM_NCCT_original_firstorder_InterquartileRange', 'HM_NCCT_original_firstorder_RobustMeanAbsoluteDeviation',
                  'HM_NCCT_original_firstorder_Entropy', 'HM_NCCT_original_firstorder_Uniformity',
                  'HM_NCCT_original_firstorder_Variance', 'HM_NCCT_original_firstorder_RobustMeanAbsoluteDeviation',
                  'HM_original_shape_Maximum2DDiameterSlice', 'HM_NCCT_original_firstorder_Mean',
                  'HM_NCCT_original_firstorder_90Percentile',
                  'HM_original_shape_MinorAxisLength', 'HM_NCCT_original_firstorder_Median',
                  'HM_NCCT_original_firstorder_Skewness', 'ED_volume', 'HM_NCCT_original_firstorder_Kurtosis',
                  'HM_original_shape_SurfaceVolumeRatio', 'ED_MCA_R_Ratio', 'HM_volume', 'HM_original_shape_VoxelVolume',
                  'HM_NCCT_original_firstorder_Maximum', 'HM_MCA_R_Ratio', 'HM_original_shape_MeshVolume',
                  'HM_original_shape_MajorAxisLength', 'HM_original_shape_SurfaceArea',
                  'HM_NCCT_original_firstorder_10Percentile', 'ED_ACA_L_Ratio', 'HM_original_shape_Flatness',
                  'HM_original_shape_Sphericity', 'HM_original_shape_Maximum3DDiameter', 'ED_PCA_R_Ratio',
                  'HM_NCCT_original_firstorder_Energy']
for tmp in select_feature:
    if tmp not in list(pd_train_feature.columns):
        print(tmp+"不在特征中")
pd_train_feature = pd_train_feature[select_feature]
pd_predict_feature = pd_predict_feature[select_feature]
'''
# data = pd_train_feature
# rho = data.corr(method='spearman')
# pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
# pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
# plt.rcParams['font.family'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(80, 60))
# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# sns.heatmap(rho, cmap=cmap, annot=True, vmin=-1).get_figure().savefig('相关性分析\全部变量相关性分析.png',
#                                                                       dpi=500,
#                                                                       bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
# # plt.show()
from collections import Counter

# 统计每个类别的数量

class_counts = dict(Counter(pd_train_label.T.values.tolist()[0]))
# 计算每个类别的权重，可以根据需要调整权重的计算方式
total_samples = len(pd_train_label.values)
class_weights = {class_label: total_samples / (class_counts[class_label] * len(class_counts)) for class_label in
                 class_counts}

X_train, X_test, Y_train, Y_test = train_test_split(pd_train_feature, pd_train_label, test_size=0.25)
X_train, X_test, Y_train, Y_test = X_train.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_test.to_numpy()
X_predict = pd_predict_feature.to_numpy()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_predict = sc.transform(X_predict)

model_class = {}

# clf1 = BalancedBaggingClassifier(random_state=42)
params2 = {
    "objective": "multi:softmax",
    "num_class": 7,
    "scale_pos_weight": class_weights,
}
clf2 = XGBClassifier(n_estimators=1000, random_state=0, n_jobs=-1, **params2)
model_class[1] = 1
clf3 = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf3.class_weight = 'balanced'
model_class[2] = 1
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 7,
    'max_depth': 300,
    'num_leaves': 500,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}  # regression
clf4 = LGB.LGBMClassifier(n_estimators=1000,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          random_state=32, **params, class_weight=class_weights)
model_class[3] = 1
clf5 = LinearRegression()
model_class[4] = 1
clf6 = ctb.CatBoostClassifier(learning_rate=0.1,
                              iterations=1,
                              depth=15,
                              random_seed=32,
                              class_weights=class_weights)
model_class[5] = 1
clf7 = KNeighborsClassifier()
model_class[6] = 1
clf8 = SVR()
model_class[7] = 1
from sklearn.neural_network import MLPRegressor, MLPClassifier

clf9 = MLPClassifier()
model_class[8] = 1
clf10 = DecisionTreeRegressor()
model_class[9] = 1
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

clf11 = AdaBoostClassifier()
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

model_class[10] = 1
clf12 = GradientBoostingClassifier()
model_class[11] = 1
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 7,
    'max_depth': 1000,
    'num_leaves': 1000,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}  # regression
clf13 = BaggingClassifier()

models = [clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf11, clf12, clf13]
clf21 = XGBClassifier(n_estimators=1000, random_state=0, n_jobs=-1, **params2)
clf31 = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf31.class_weight = 'balanced'
model_class[2] = 1

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 7,
    'max_depth': 300,
    'num_leaves': 500,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}  # regression
clf41 = LGB.LGBMClassifier(n_estimators=1000,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           random_state=32, **params, class_weight=class_weights)
clf51 = LinearRegression()
clf61 = ctb.CatBoostClassifier(learning_rate=0.1,
                               iterations=1,
                               depth=15,
                               random_seed=32)

clf71 = KNeighborsClassifier()
clf81 = SVR()

clf91 = MLPRegressor()
clf101 = DecisionTreeRegressor()
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

clf111 = AdaBoostClassifier()
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

clf121 = GradientBoostingClassifier()
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

clf131 = BaggingClassifier()

models1 = [clf21, clf31, clf41, clf51, clf61, clf71, clf81, clf91, clf101, clf111, clf121, clf131]

loss = []
T_loss = []
f1_s = []
model_name = []
## 单独模型训练

for i, model in enumerate(models):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_pred1 = model.predict(X_train)
    y_ans = model.predict(X_predict)
    if model_class[i + 1] == 0:
        y_pred = y_pred.argmax(axis=1)
        y_pred1 = y_pred1.argmax(axis=1)
        y_ans = y_ans.argmax(axis=1)
    MAE_tmp = metrics.mean_absolute_error(Y_test, np.round(y_pred))
    title_name = str(model)[0:10]
    title_name = re.sub(r'[\/:*?"<>|]', ' ', title_name)
    model_name.append(title_name)
    loss.append(metrics.mean_absolute_error(Y_test, np.round(y_pred)))
    T_loss.append(metrics.mean_absolute_error(Y_train, np.round(y_pred1)))
    f1_s.append(metrics.f1_score(Y_test, np.round(y_pred), average='weighted'))
    print('11111测试集SAE:', metrics.mean_absolute_error(Y_test, np.round(y_pred)))
    plt.figure()
    plt.subplot(2, 1, 1)

    plt.title(str(model)[0:10])
    plt.scatter(range(len(X_test)), y_pred)
    plt.scatter(range(len(X_test)), Y_test)
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(X_train)), y_pred1)
    plt.scatter(range(len(X_train)), Y_train)
    plt.savefig("堆叠模型预测/" + title_name + "模型预测图.png")
    pd.DataFrame(np.round(y_ans)).to_csv(r"堆叠模型预测/" + title_name + "模型mRS.csv", mode='w', header=True,
                                         encoding='gb18030')
    plt.show()

pd.DataFrame(loss).to_excel(r'堆叠模型预测\单模型误差loss.xlsx')
pd.DataFrame(T_loss).to_excel(r'堆叠模型预测\单模型误差T_loss.xlsx')
pd.DataFrame(f1_s).to_excel(r'堆叠模型预测\单模型误差f1_s.xlsx')
pd.DataFrame(model_name).to_excel(r'堆叠模型预测\单模型名称.xlsx')

from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import RidgeCV

## 多模型堆叠训练
stackmodels = []
sortloss = []
sort_T_loss = []
for i in range(len(models)):
    minindex = loss.index(min(loss))
    stackmodels.append(models1[minindex])
    sortloss.append(loss[minindex])
    sort_T_loss.append(T_loss[minindex])
    loss.pop(minindex)
    models1.pop(minindex)
print(stackmodels)
print(sortloss)
# 第二层模型
estimatorses = []
stackmodels_loss = []
stackmodels_Tloss = []
stackmodels_f1 = []
print(Y_train)

for n in range(8):
    estimators = []
    for m in range(n + 1):
        model = stackmodels[m]
        name = 'aaa' + str(m)
        estimators.append((name, model))
    print(estimators)
    estimatorses.append(estimators)
    sclf = StackingClassifier(
        estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    )
    print(Y_train.shape)
    sclf.fit(X_train, Y_train)

    # 模型评估
    from sklearn import metrics
    import seaborn as sns
    from sklearn.metrics import f1_score

    y_pred = sclf.predict(X_test)  # 预测类别
    y_pred1 = sclf.predict(X_train)  # 预测类别
    MAE_tmp = metrics.mean_absolute_error(Y_test, np.round(y_pred))
    title_name = "堆叠" + str(model)[0:10] + ":" + str(MAE_tmp)[0:5]
    title_name = re.sub(r'[\/:*?"<>|]', ' ', title_name)

    print("堆叠模型训练##########################################################")
    # print(classification_report(Y_test, round_score(predict_results)))
    print('测试集MSE:', metrics.mean_squared_error(Y_test, y_pred))
    print('测试集MAE:', metrics.mean_absolute_error(Y_test, y_pred))
    print('训练集MAE:', metrics.mean_absolute_error(Y_train, y_pred1))

    plt.figure()
    plt.subplot(2, 1, 1)

    plt.title("堆叠" + str(model)[0:10])
    plt.scatter(range(len(X_test)), y_pred)
    plt.scatter(range(len(X_test)), Y_test)
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(X_train)), y_pred1)
    plt.scatter(range(len(X_train)), Y_train)
    plt.savefig("堆叠模型预测/" + title_name + "模型预测图.png")
    plt.show()

    stackmodels_loss.append(metrics.mean_absolute_error(Y_test, y_pred))
    stackmodels_Tloss.append(metrics.mean_absolute_error(Y_train, y_pred1))

pd.DataFrame(stackmodels_loss).to_excel(r'堆叠模型预测\堆叠模型误差loss.xlsx')
pd.DataFrame(stackmodels_Tloss).to_excel(r'堆叠模型预测\堆叠模型误差T_loss.xlsx')

print(stackmodels_loss)
print(estimatorses)

minindex = stackmodels_loss.index(min(stackmodels_loss))
print(stackmodels_loss[minindex])
print(stackmodels_Tloss[minindex])
print('=============================================')
print('最终堆叠模型基学习器数量', len(estimatorses[minindex]))
print('最终堆叠模型基学习器', estimatorses[minindex])

sclf = StackingClassifier(
    estimators=estimatorses[minindex],
    final_estimator=RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
)

sclf.fit(X_train, Y_train)
y_pred = sclf.predict(X_test)  # 预测类别
y_pred1 = sclf.predict(X_train)  # 预测类别
print('测试集MAE:', metrics.mean_absolute_error(Y_test, np.round(y_pred)))
Y_test1 = []
result = sclf.predict(X_predict)  #
pd.DataFrame(np.round(result)).to_csv(r"堆叠模型预测\最终预测mRS.csv", mode='w', header=True, encoding='gb18030')
