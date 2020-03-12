"""
A demo to plot beautiful graphs

"""
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")


csv_path = r'D:\PAPER_LATEX\2020_paper\eccv2018kit_kaggle\figures\bayesian_weights'

csv1 = pd.read_csv(os.path.join(csv_path, 'carcls1.csv'))
csv2 = pd.read_csv(os.path.join(csv_path, 'carcls2.csv'))
csv3 = pd.read_csv(os.path.join(csv_path, 'carcls3.csv'))
carcls_csv = pd.concat([csv1, csv2, csv3])
carcls_csv.columns = ['~', 'steps', 'CarCls']
carcls_csv = carcls_csv.drop('~', 1)





csv1 = pd.read_csv(os.path.join(csv_path, 'rot1.csv'))
csv2 = pd.read_csv(os.path.join(csv_path, 'rot2.csv'))
csv3 = pd.read_csv(os.path.join(csv_path, 'rot3.csv'))
rot_csv = pd.concat([csv1, csv2, csv3])
rot_csv.columns = ['~', 'steps', 'Rot']
rot_csv = rot_csv.drop('~', 1)


csv1 = pd.read_csv(os.path.join(csv_path, 'trans1.csv'))
csv2 = pd.read_csv(os.path.join(csv_path, 'trans2.csv'))
csv3 = pd.read_csv(os.path.join(csv_path, 'trans3.csv'))
trans_csv = pd.concat([csv1, csv2, csv3])
trans_csv.columns = ['~', 'steps', 'Trans']
trans_csv = trans_csv.drop('~', 1)

df = pd.merge(trans_csv, pd.merge(carcls_csv, rot_csv, on='steps'), on='steps')

plt.plot(df['steps'], df['CarCls'])
plt.plot(df['steps'], df['Rot'])
plt.plot(df['steps'], df['Trans'])
plt.legend(['CarCls', 'Rot', 'Trans'], loc='upper left')
plt.xlabel('steps')
plt.ylabel('weights')
plt.savefig(os.path.join(csv_path, 'all.png'))
plt.show()
#
#
# sns_plot = sns.lineplot(x="steps", y="CarCls", data=carcls_csv, color='coral', linewidth=3)
# fig = sns_plot.get_figure()
# fig.savefig(os.path.join(csv_path, 'carcls.png'))
# plt.show()
#
#
# sns_plot = sns.lineplot(x="steps", y="Rot", data=rot_csv, color='violet', linewidth=3)
# fig = sns_plot.get_figure()
# fig.savefig(os.path.join(csv_path, 'rot.png'))
# plt.show()
#
#
# sns_plot = sns.lineplot(x="steps", y="Trans", data=trans_csv, linewidth=3)
# fig = sns_plot.get_figure()
# fig.savefig(os.path.join(csv_path, 'tran.png'))
# plt.show()