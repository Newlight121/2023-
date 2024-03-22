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

time_header = table_ex1.head(0)
## issue1
mem_time = {}
happen_time = {}
happen_value = {}
ED_time = {}
ED_Value = {}

for index, line in (table1["入院首次影像检查流水号"]).iteritems():
    mem_time[index] = 0
    happen_time[index] = 0
    happen_value[index] = 0
    init_time_interval = table1.loc[index]["发病到首次影像检查时间间隔"]

    table2_num = table2.loc[index]["首次检查流水号"]  # 记录首次的HM
    if table2_num == line:
        init_HM = table2.loc[index]["HM_volume"]
    else:
        print("表2与表1对应错误", index, line)

    continue_time = table_ex1.loc[index]  # 获取sub对应的随访记录次数
    init_time = continue_time["入院首次检查时间点"]  # 获取首次检查时间点
    ensure_num = continue_time["入院首次检查流水号"]  # 获取首次检查时间点
    if ensure_num == line:
        re_times = continue_time["重复次数"]
        for i in range(re_times - 1):
            add_time = continue_time["随访" + str(i + 1) + "时间点"]  # 获取随访时间点
            add_num = continue_time["随访" + str(i + 1) + "流水号"]  # 获取随访流水号
            try:
                add_HM = table2.loc[index]["HM_volume." + str(i + 1)]  # 记录随访的HM
            except:
                print("随访次数错误", index, line, re_times)

            cal_time = (add_time - init_time).days * 24 + (
                    add_time - init_time).seconds / 3600 + init_time_interval  # 计算总小时数量
            cal_HM = add_HM - init_HM  # 增加体积
            if cal_time < 48:
                if ((cal_HM >= 6000) or (cal_HM / init_HM >= 0.33)):
                    mem_time[index] = 1
                    happen_time[index] = cal_time
                    happen_value[index] = add_HM
                    break

    else:
        print("附件1与表1对应错误", index, line)
pass

pd.DataFrame(mem_time.values()).to_excel("是否发生血肿.xlsx")
pd.DataFrame(happen_time.values()).to_excel("发生血肿时间.xlsx")

HM_time = {}
HM_Value = {}
ED_time = {}
ED_Value = {}

for index, line in (table1["入院首次影像检查流水号"]).iteritems():
    init_time_interval = table1.loc[index]["发病到首次影像检查时间间隔"]
    HM_time[index] = [init_time_interval]
    HM_Value[index] = [table2.loc[index]["HM_volume"]]
    ED_time[index] = [init_time_interval]
    ED_Value[index] = [table2.loc[index]["ED_volume"]]

    table2_num = table2.loc[index]["首次检查流水号"]  # 记录首次的HM
    if table2_num == line:
        init_HM = table2.loc[index]["HM_volume"]
    else:
        print("表2与表1对应错误", index, line)

    continue_time = table_ex1.loc[index]  # 获取sub对应的随访记录次数
    init_time = continue_time["入院首次检查时间点"]  # 获取首次检查时间点
    ensure_num = continue_time["入院首次检查流水号"]  # 获取首次检查时间点
    if ensure_num == line:
        re_times = continue_time["重复次数"]
        for i in range(re_times - 1):
            add_time = continue_time["随访" + str(i + 1) + "时间点"]  # 获取随访时间点
            add_num = continue_time["随访" + str(i + 1) + "流水号"]  # 获取随访流水号
            try:
                add_HM = table2.loc[index]["HM_volume." + str(i + 1)]  # 记录随访的HM
            except:
                print("随访次数错误", index, line, re_times)

            cal_time = (add_time - init_time).days * 24 + (
                    add_time - init_time).seconds / 3600 + init_time_interval  # 计算总小时数量
            cal_HM = add_HM - init_HM

            add_ED = table2.loc[index]["ED_volume." + str(i + 1)]  # 记录随访的ED
            ED_time[index].append(cal_time)  # 记录水肿数据时间
            ED_Value[index].append(add_ED)
            HM_time[index].append(cal_time)  # 记录水肿数据时间
            HM_Value[index].append(add_HM)
    else:
        print("附件1与表1对应错误", index, line)
pass

df_ishappen = pd.DataFrame.from_dict(mem_time, orient='index', columns=['是否发生血肿'])
df_happentime = pd.DataFrame.from_dict(happen_time, orient='index', columns=['发生血肿时间'])
df_happenvalue = pd.DataFrame.from_dict(happen_value, orient='index', columns=['血肿大小'])
df_ishappen.to_excel("血肿记录绘图\是否发生血肿.xlsx")
df_happentime.to_excel("血肿记录绘图\发生血肿时间.xlsx")
df_happenvalue.to_excel("血肿记录绘图\血肿发生时大小.xlsx")

import matplotlib.ticker as ticker

'''
for index, line in (df_ishappen["是否发生血肿"]).items():
    if line == 1:  # 发生血肿
        fig, ax = plt.subplots(figsize=(7, 3), dpi=500)
        ax.plot(HM_time[index], HM_Value[index], marker="o", mfc="white", ms=5, zorder=1)
        ax.fill_between(x=HM_time[index], y1=HM_Value[index], y2=0, alpha=0.5)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.scatter(df_happentime.loc[index].values, df_happenvalue.loc[index].values, marker="o", c='r',s=50, zorder=5)
        # plt.show()
        plt.savefig("血肿记录/"+index+"血肿记录.png")
'''
# df_happentime.to_excel("是否发生血肿.xlsx")
# df_happentime.to_excel("发生血肿时间.xlsx")

df_ED_time = pd.DataFrame.from_dict(ED_time, orient='index')
df_ED_Value = pd.DataFrame.from_dict(ED_Value, orient='index')
df_HM_time = pd.DataFrame.from_dict(HM_time, orient='index')
df_HM_Value = pd.DataFrame.from_dict(HM_Value, orient='index')

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

## 提取table3首次记录
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

# df_table3_extract = pd.DataFrame.from_dict(table3_extract, orient='index', columns=table3_feature_name)
df_table3_hm_extract = pd.DataFrame.from_dict(table3_HM_extract, orient='index', columns=table3_feature_name)
df_table3_ed_extract = pd.DataFrame.from_dict(table3_ED_extract, orient='index', columns=table3_feature_name)
df_table3_hm_extract = df_table3_hm_extract.rename(columns=lambda x: "HM_" + x)
df_table3_ed_extract = df_table3_ed_extract.rename(columns=lambda x: "ED_" + x)
train_index = ['sub' + str(i + 1).zfill(3) for i in range(100)]
predict_index_tmp = ['sub' + str(i + 101).zfill(3) for i in range(60)]
# predict_index_tmp.remove('sub131')
# predict_index_tmp.remove('sub132')
'''
predict_index = train_index + predict_index_tmp

table1_feature = table1[table1_feature_name]
table2_feature = table2[table2_feature_name]

table1_feature_train = table1_feature.loc[train_index]
table2_feature_train = table2_feature.loc[train_index]
table3_feature_train = df_table3_extract.loc[train_index]

table1_feature_predict = table1_feature.loc[predict_index]
table2_feature_predict = table2_feature.loc[predict_index]
table3_feature_predict = df_table3_extract.loc[predict_index]
# table3_feature = table3[table3_feature_name]
# table1_feature =

# 连接训练集与预测集
pd_tmp = pd.merge(table1_feature_train, table2_feature_train, how='left', left_index=True, right_index=True)
pd_train_feature = pd.merge(pd_tmp, table3_feature_train, how='left', left_index=True, right_index=True)
pd_tmp = pd.merge(table1_feature_predict, table2_feature_predict, how='left', left_index=True, right_index=True)
pd_train_predict = pd.merge(pd_tmp, table3_feature_predict, how='left', left_index=True, right_index=True)
'''

predict_index = train_index + predict_index_tmp
pd_train_label = df_ishappen.loc[train_index]
pd_predict_label = df_ishappen.loc[predict_index]

table1_feature = table1[table1_feature_name]
table2_feature = table2[table2_feature_name]

table_feature = pd.merge(table1_feature, table2_feature, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, df_table3_hm_extract, how='left', left_index=True,
                         right_index=True)
table_feature = pd.merge(table_feature, df_table3_ed_extract, how='left', left_index=True,
                         right_index=True)

pd_train_feature = table_feature.loc[train_index]
pd_predict_feature = table_feature.loc[predict_index]

from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

m1 = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='binary:logistic')
# m1 = RandomForestRegressor(n_estimators=1500, max_features='log2')
m1.class_weight = 'balanced'
# 训练模型

m1.fit(pd_train_feature.to_numpy(), pd_train_label.to_numpy())
m1.get_booster().feature_names = list(pd_train_feature.columns)
importance_values = m1.get_booster().get_score(importance_type="gain")
pd.DataFrame.from_dict(importance_values, orient='index').to_excel("所有变量重要性顺序.xlsx")
# feature_value = m1.feature_importances_
feature_name, feature_value = list(importance_values.keys()), list(importance_values.values())
# feature_name = pd_train_feature.columns
sorted_data = sorted(zip(feature_name, feature_value), key=lambda x: x[1], reverse=True)
feature_name, feature_value = zip(*sorted_data)
plt.figure(figsize=(15, 20))  # 设置图像大小
plt.barh(feature_name, feature_value, color='skyblue')
plt.gca().invert_yaxis()
plt.yticks(fontsize=10)
plt.title('变量重要性排序', fontsize=25)
plt.tight_layout()
plt.savefig("相关性分析\全变量特征重要性.png", dpi=500)
plt.show()

import seaborn as sns

# 斯皮尔曼相关系数（Spearman correlation）,一般来说要剔除相关性过于高的变量，因为可以认为他们重复了-------------------------------
data = pd.merge(pd_train_feature, pd_train_label, how='left', left_index=True, right_index=True).copy()

discrete_feature = ['性别', '脑出血前mRS评分', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史', '吸烟史',
                    '饮酒史', '脑室引流', '止血治疗', '降颅压治疗', '降压治疗',
                    '镇静、镇痛治疗', '止吐护胃', '营养神经']
feature = data.columns.tolist()
continues_feature = [x for x in feature if x not in discrete_feature]
discrete_feature.append('是否发生血肿')

continues_data = data[continues_feature]
discrete_data = data[discrete_feature]
'''
data = continues_data
rho = data.corr(method='spearman')
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(40, 35))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(rho, cmap=cmap, annot=True, vmin=-1).get_figure().savefig('相关性分析\连续变量血肿相关性分析.png',
                                                                      dpi=800,
                                                                      bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
plt.show()
high_corr1 = abs(rho.loc["是否发生血肿"]).sort_values(ascending=False)

high_corr_show = rho[high_corr1[0:12].index].loc[high_corr1[0:12].index]
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(16, 12))
sns.heatmap(high_corr_show, cmap=cmap, annot=True, vmin=-1).get_figure().savefig(
    '相关性分析\连续变量血肿相关性分析_显示少部分.png', dpi=500,
    bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
plt.show()

data = discrete_data
rho = data.corr(method='kendall')
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 12))
sns.heatmap(rho, cmap=cmap, annot=True, vmin=-1).get_figure().savefig('相关性分析\离散变量相关性分析.png',
                                                                      dpi=500,
                                                                      bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
plt.show()
high_corr2 = abs(rho.loc["是否发生血肿"]).sort_values(ascending=False)

high_corr1.to_excel('相关性分析\连续变量相关性排序.xlsx')
high_corr2.to_excel('相关性分析\离散变量相关性排序.xlsx')
high_corr1[1:12].index
high_corr2[1:8].index
'''
high_corr_name = ['original_shape_Elongation', 'ED_PCA_R_Ratio', 'ED_Cerebellum_R_Ratio',
                  '发病到首次影像检查时间间隔', 'NCCT_original_firstorder_Variance', 'HM_PCA_R_Ratio',
                  'NCCT_original_firstorder_RobustMeanAbsoluteDeviation',
                  'NCCT_original_firstorder_InterquartileRange',
                  'NCCT_original_firstorder_MeanAbsoluteDeviation',
                  'NCCT_original_firstorder_Range', 'HM_ACA_R_Ratio', '冠心病史', '房颤史', '饮酒史', '止血治疗',
                  '吸烟史', '降颅压治疗', '卒中病史']

pd_train_high_feature = pd_train_feature
pd_predict_high_feature = pd_predict_feature
# pd_train_high_feature = pd_train_feature[high_corr_name]
# pd_predict_high_feature = pd_train_predict[high_corr_name]
# dsc = pd_train_high_feature.describe()
# pd.plotting.scatter_matrix(pd_train_high_feature, figsize=(20,10), alpha=0.75)
# plt.show()
# pd_train_label.sum()


use_regree = False
lowest_rms = 0
best_weight = 0
pred_ans_mem = 0
roc_mem = 0
best_acc = {"precision": 0, "recall": 0, "f1_score": 0}
for i in range(1, 500):
    weight_given = 0.01 * i  # 权重分布
    # weight_given = 2.17
    pd_weight = pd_train_label.copy().rename(columns={'是否发生血肿': '权重'})
    for index, line in pd_train_label["是否发生血肿"].items():
        if line == 1:
            pd_weight.loc[index] = weight_given
        elif line == 0:
            pd_weight.loc[index] = 1

    pd_train_high_feature_weight = pd.merge(pd_train_high_feature, pd_weight, how='left', left_index=True,
                                            right_index=True).copy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import lightgbm as LGB
    from sklearn import metrics

    X_predict = pd_predict_high_feature.to_numpy()
    Y_predict = pd_predict_label.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(pd_train_high_feature_weight, pd_train_label, test_size=0.25)
    train_weight = X_train['权重'].to_numpy()  # 获取权重
    X_train.drop(labels='权重', axis=1, inplace=True)  # 删除权重列
    X_test.drop(labels='权重', axis=1, inplace=True)
    X_train, X_test, Y_train, Y_test = X_train.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_test.to_numpy()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_predict = sc.transform(X_predict)

    if use_regree:
        # 创建随机森林回归模型
        m1 = RandomForestRegressor(n_estimators=1500, max_features='log2')
        m1.class_weight = 'balanced'
        # 训练模型
        m1.fit(X_train, Y_train)
    else:
        # 建立LGB的dataset格式数据
        lgb_train = LGB.Dataset(X_train, Y_train, weight=train_weight)
        lgb_eval = LGB.Dataset(X_test, Y_test, reference=lgb_train)
        # 定义超参数dict
        evals_result = {}
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_class': 1,
            'max_depth': 1000,
            'num_leaves': 1000,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }  # regression
        # 定义callback回调
        callback = [LGB.early_stopping(stopping_rounds=10, verbose=True),
                    LGB.log_evaluation(period=10, show_stdv=True)]
        # LGB训练 train
        m1 = LGB.train(params, lgb_train, num_boost_round=2000,
                       valid_sets=[lgb_train, lgb_eval], callbacks=callback)
    y_pred1 = m1.predict(X_train)
    y_pred = m1.predict(X_test)
    y_pred_ans = m1.predict(X_predict)
    # plt.subplot(2, 1, 1)
    # plt.scatter(range(len(X_test)), y_pred)
    # plt.scatter(range(len(X_test)), Y_test)
    # plt.subplot(2, 1, 2)
    # plt.scatter(range(len(X_train)), y_pred1)
    # plt.scatter(range(len(X_train)), Y_train)
    # plt.show()
    # y_pred = y_pred.argmax(axis=1)
    RMSE = metrics.mean_squared_error(Y_train, (y_pred1))
    F1traintmp = metrics.f1_score(Y_train, np.round(y_pred1), average='binary')
    F1testtmp = metrics.f1_score(Y_test, np.round(y_pred), average='binary')
    recall_tmp = metrics.recall_score(Y_test, np.round(y_pred), average='binary')
    precision_tmp = metrics.precision_score(Y_test, np.round(y_pred))
    MAE_test = metrics.mean_squared_error(Y_test, (y_pred))
    if (F1testtmp + 1 - MAE_test) > lowest_rms:
        lowest_rms = (F1testtmp + 1 - MAE_test)
        print("更新F1", (F1testtmp + 1 - MAE_test))
        best_weight = weight_given
        best_acc["precision"] = precision_tmp
        best_acc["recall"] = recall_tmp
        best_acc["f1_score"] = F1testtmp
        pred_ans_mem = m1.predict(X_predict)

        # pd.DataFrame(pred_ans_mem).to_excel("保存血肿结果6.xlsx")

## issue2
table1 = pd.read_excel("表1-患者列表及临床信息.xlsx", index_col=0)
# table1_100 = pd.read_excel("表1-前100.xlsx", index_col=0)
table2 = pd.read_excel("表2-患者影像信息血肿及水肿的体积及位置.xlsx", index_col=0)
table3_ED = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="ED")
table3_Hemo = pd.read_excel("表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx", sheet_name="Hemo")
table4 = pd.read_excel("表4-答案文件.xlsx")
table_ex1 = pd.read_excel("附表1-检索表格-流水号vs时间.xlsx", index_col=0)

ED_time_100 = df_ED_time.loc[train_index]
ED_value_100 = df_ED_Value.loc[train_index]
HM_time_100 = df_HM_time.loc[train_index]
HM_value_100 = df_HM_Value.loc[train_index]
# ED_time_100.to_excel("ED time.xlsx")
# ED_value_100.to_excel("ED value.xlsx")
HM_time_100.to_excel("HM time.xlsx")
HM_value_100.to_excel("HM value.xlsx")

# ED_time_dict_100 = ED_time_100.to_dict(orient='index')
ED_time_dict_100 = {key: ED_time[key] for key in train_index}
ED_value_dict_100 = {key: ED_Value[key] for key in train_index}
ED_count = {key: len(ED_Value[key]) for key in train_index}
df_ED_count = pd.DataFrame.from_dict(ED_count, orient='index')
df_ED_count.to_excel("ED Count.xlsx")
x = list(ED_time_dict_100.values())
y = list(ED_value_dict_100.values())

df_ED_time_copy = df_ED_time.copy()
df_ED_Value_copy = df_ED_Value.copy()
# max_count = df_ED_count.max().values # 按照最多曲线数量进行插值
max_count = 8
df_ED_time_insert = pd.DataFrame(index=df_ED_time.index, columns=range(max_count), data=np.nan)
df_ED_Value_insert = pd.DataFrame(index=df_ED_Value.index, columns=range(max_count), data=np.nan)
for ED_index, ED_num in (df_ED_count[0]).items():
    insert_count = max_count - ED_num
    df_ed_tmp = df_ED_time.loc[ED_index].dropna(how='all').to_frame()  # 取对应行
    df_ed_value_tmp = df_ED_Value.loc[ED_index].dropna(how='all').to_frame()  # 取对应行
    tmp = list(df_ed_tmp.values)  # list
    for k in range(int(insert_count)):  # 插值次数
        max_inter, start_mem = 0, 0
        for i, start in enumerate(range(len(tmp) - 1)):
            inter = tmp[i + 1] - tmp[i]
            if inter > max_inter:
                max_inter = int(inter)
                start_mem = i + 1  # 记录需要插值位置
        df_ed_tmp = pd.DataFrame(np.insert(df_ed_tmp.values, start_mem, values=np.nan, axis=0))
        df_ed_value_tmp = pd.DataFrame(np.insert(df_ed_value_tmp.values, start_mem, values=np.nan, axis=0))
        df_ed_tmp, df_ed_value_tmp = df_ed_tmp.interpolate(), df_ed_value_tmp.interpolate()
        tmp = list(df_ed_tmp.values)  # list
        # df_tmp.insert(loc=start_mem, value=np.nan)
    df_ED_time_insert.loc[ED_index] = df_ed_tmp.iloc[:, 0]
    df_ED_Value_insert.loc[ED_index] = df_ed_value_tmp.iloc[:, 0]

df_ED_time_insert1 = df_ED_time_insert.loc[train_index]
df_ED_Value_insert1 = df_ED_Value_insert.loc[train_index]
df_ED_time_insert1.to_excel("ED time insert-" + str(max_count) + ".xlsx")
df_ED_Value_insert1.to_excel("ED value insert-" + str(max_count) + ".xlsx")

# HM_time_dict_100 = HM_time_100.to_dict(orient='index')
HM_time_dict_100 = {key: HM_time[key] for key in train_index}
HM_value_dict_100 = {key: HM_Value[key] for key in train_index}
HM_count = {key: len(HM_Value[key]) for key in train_index}
df_HM_count = pd.DataFrame.from_dict(HM_count, orient='index')
df_HM_count.to_excel("HM Count.xlsx")
x = list(HM_time_dict_100.values())
y = list(HM_value_dict_100.values())

df_HM_time_copy = df_HM_time.copy()
df_HM_Value_copy = df_HM_Value.copy()
# max_count = df_HM_count.max().values # 按照最多曲线数量进行插值
max_count = 8
df_HM_time_insert = pd.DataFrame(index=df_HM_time.index, columns=range(max_count), data=np.nan)
df_HM_Value_insert = pd.DataFrame(index=df_HM_Value.index, columns=range(max_count), data=np.nan)
for HM_index, HM_num in (df_HM_count[0]).items():
    insert_count = max_count - HM_num
    df_HM_tmp = df_HM_time.loc[HM_index].dropna(how='all').to_frame()  # 取对应行
    df_HM_value_tmp = df_HM_Value.loc[HM_index].dropna(how='all').to_frame()  # 取对应行
    tmp = list(df_HM_tmp.values)  # list
    for k in range(int(insert_count)):  # 插值次数
        max_inter, start_mem = 0, 0
        for i, start in enumerate(range(len(tmp) - 1)):
            inter = tmp[i + 1] - tmp[i]
            if inter > max_inter:
                max_inter = int(inter)
                start_mem = i + 1  # 记录需要插值位置
        df_HM_tmp = pd.DataFrame(np.insert(df_HM_tmp.values, start_mem, values=np.nan, axis=0))
        df_HM_value_tmp = pd.DataFrame(np.insert(df_HM_value_tmp.values, start_mem, values=np.nan, axis=0))
        df_HM_tmp, df_HM_value_tmp = df_HM_tmp.interpolate(), df_HM_value_tmp.interpolate()
        tmp = list(df_HM_tmp.values)  # list
        # df_tmp.insert(loc=start_mem, value=np.nan)
    df_HM_time_insert.loc[HM_index] = df_HM_tmp.iloc[:, 0]
    df_HM_Value_insert.loc[HM_index] = df_HM_value_tmp.iloc[:, 0]

df_HM_time_insert1 = df_HM_time_insert.loc[train_index]
df_HM_Value_insert1 = df_HM_Value_insert.loc[train_index]
df_HM_time_insert1.to_excel("HM time insert-" + str(max_count) + ".xlsx")
df_HM_Value_insert1.to_excel("HM value insert-" + str(max_count) + ".xlsx")

# 以two_moons数据为例
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# 生成模拟的二维数据, X.shape——>(100, 2)
# 设置为三个聚类中心
# 提取特征列
ED_fit_par = pd.read_excel("ED fit par-linemuxmax.xlsx", header=None);
features = ED_fit_par  # 权重
# 使用Min-Max标准化对特征进行权重归一化
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

n_clusters = 5
Kmeans = KMeans(n_clusters)
kmeans_labels = Kmeans.fit_predict(scaled_features)

gmm = GaussianMixture(n_clusters)
gmm.fit(scaled_features)
gmm_labels = gmm.predict(scaled_features)

# 使用T-SNE进行降维可视化
tsne = TSNE(n_components=2)  # 降维到2维
tsne_data = tsne.fit_transform(scaled_features)
# 创建散点图可视化T-SNE结果
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=gmm_labels)
plt.title('T-SNE GMM')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.savefig("T-SNE GMM降维结果.png")
plt.show()

# 使用T-SNE进行降维可视化
tsne = TSNE(n_components=2)  # 降维到2维
tsne_data = tsne.fit_transform(scaled_features)
# 创建散点图可视化T-SNE结果
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels)
plt.title('T-SNE Kmeans')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.savefig("T-SNE Kmeans降维结果.png")
plt.show()

# 输出K-Means的聚类结果
print("K-Means Clusters:")
print(kmeans_labels)
kmeans_value = pd.DataFrame(index=df_ED_time_insert1.index, columns=range(1), data=kmeans_labels)
kmeans_value.to_excel("Kmeans聚类.xlsx")

print("GMM Clusters:")
print(gmm_labels)
gmm_value = pd.DataFrame(index=df_ED_time_insert1.index, columns=range(1), data=gmm_labels)
gmm_value.to_excel("GMM聚类.xlsx")

for cluster in range(n_clusters):
    df_ED_time_insert1.loc[kmeans_value.values == cluster].to_excel(
        "Kmeans聚类_cluster" + str(cluster) + "ED Time.xlsx")
    df_ED_Value_insert1.loc[kmeans_value.values == cluster].to_excel(
        "Kmeans聚类_cluster" + str(cluster) + "ED Value.xlsx")
    df_ED_time_insert1.loc[gmm_value.values == cluster].to_excel("GMM聚类_cluster" + str(cluster) + "ED Time.xlsx")
    df_ED_Value_insert1.loc[gmm_value.values == cluster].to_excel("GMM聚类_cluster" + str(cluster) + "ED Value.xlsx")

# issue2-3
cluster_result = gmm_value.rename(columns={0: '聚类结果'})
measure = ['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']
pd_measure = table1[measure].loc[train_index]
data = pd.merge(cluster_result, pd_measure, how='left', left_index=True, right_index=True).copy()
rho = data.corr(method='kendall')
pd.set_option('display.max_rows', None)  # 可以填数字，填None表示'行'无限制
pd.set_option('display.max_columns', None)  # 可以填数字，填None表示'列'无限制

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(15, 10))
# sns.heatmap(rho,
#             cmap='Reds',
#             annot=True,
#             # fmt='d'
#             ).get_figure().savefig('output.png', dpi=500, bbox_inches='tight')  # fmt显示完全，dpi显示清晰，bbox_inches保存完全
# plt.show()

# XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 训练模型
# model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='binary:logistic')
m1 = RandomForestRegressor(n_estimators=1500, max_features='log2')
m1.class_weight = 'balanced'
# 训练模型
m1.fit(X_train, Y_train)
# m1.get_booster().feature_names = measure
feature_value = m1.feature_importances_
feature_name = pd_measure.columns
# importance_values = model.get_booster().get_score(importance_type="gain")
# # importance_values = model.get_booster().get_score(importance_type="gain")
# feature_name = list(importance_values.keys())
# feature_value = list(importance_values.values())
# 创建直方图
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.bar(feature_name, feature_value, color='skyblue')
plt.title('重要性分布')
plt.ylabel('重要性')
# 隐藏x轴标签，使得序列名称不会重叠
plt.xticks(rotation=45, ha='right')
# 显示图像
plt.tight_layout()
plt.savefig("SPSS卡方分析\特征重要性.png")
plt.show()

import pandas as pd

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
pd_train_label = table1[mRS_name].loc[train_index]
pd_predict_feature = table_feature.loc[predict_index]

use_regree = True
X_train, X_test, Y_train, Y_test = train_test_split(pd_train_feature, pd_train_label, test_size=0.25)
# train_weight = X_train['权重'].to_numpy()  # 获取权重
# X_train.drop(labels='权重', axis=1, inplace=True)  # 删除权重列
# X_test.drop(labels='权重', axis=1, inplace=True)
X_train, X_test, Y_train, Y_test = X_train.to_numpy(), X_test.to_numpy(), Y_train.to_numpy(), Y_test.to_numpy()
X_predict = pd_predict_feature.to_numpy()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_predict = sc.transform(X_predict)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

if use_regree:
    # 创建随机森林回归模型
    m1 = RandomForestClassifier(n_estimators=1500, max_features='log2')
    m1.class_weight = 'balanced'
    # 训练模型
m1.fit(X_train, Y_train)
y_pred1 = m1.predict(X_train)
y_pred = m1.predict(X_test)
y_pred_ans = m1.predict(X_predict)
plt.subplot(2, 1, 1)
plt.scatter(range(len(X_test)), y_pred)
plt.scatter(range(len(X_test)), Y_test)
plt.subplot(2, 1, 2)
plt.scatter(range(len(X_train)), y_pred1)
plt.scatter(range(len(X_train)), Y_train)
plt.show()
y_pred = y_pred.argmax(axis=1)
RMSE = metrics.mean_squared_error(Y_train, (y_pred1))
F1traintmp = metrics.f1_score(Y_train, np.round(y_pred1), average='binary')
F1testtmp = metrics.f1_score(Y_test, np.round(y_pred), average='binary')
recall_tmp = metrics.recall_score(Y_test, np.round(y_pred), average='binary')
precision_tmp = metrics.precision_score(Y_test, np.round(y_pred))

person_first = table1[person_index].loc[train_index]
m1 = RandomForestRegressor(n_estimators=1500, max_features='log2')
m1.class_weight = 'balanced'
