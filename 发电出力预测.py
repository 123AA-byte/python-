import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler#归一化
import warnings
warnings.filterwarnings('ignore')#忽略警告
#设置随机种子以确保结果可复现
torch.manual_seed(42)#设置CPU的随机种子
np.random.seed(42)#设置numpy的随机种子
#设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_style('whitegrid')
#设置图表图例文字为英文
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


#数据准备
#生成模拟电力负荷数据
def generate_power_data(hours=2000):
    #生成时间序列
    dates=pd.date_range(start='2020-01-01',periods=hours,freq='H')#参数的意思分别为：起始时间，时间间隔，freq表示频率（小时）
    #基础负荷模式（日周期性）
    hour_of_day=np.array([d.hour for d in dates])
    #日负荷模式（双峰：早晨和傍晚）
    daily_pattern=100+50*np.sin(2*np.pi*hour_of_day/24)+\
    30*np.sin(4*np.pi*hour_of_day/24)
    #周负荷模式（工作日VS周末）
    day_of_week=np.array([d.weekday() for d in dates])#d.weekday()返回星期几，0-6分别代表周一到周日
    weekly_pattern=np.where(day_of_week<5,1.0,0.8)#工作日负荷为1.0，周末负荷为0.8
    #季节性模式
    day_of_year=np.array([d.timetuple().tm_yday for d in dates])
    seasonal_pattern=1+0.2*np.sin(2*np.pi*day_of_year/365-np.pi/2)
    #温度影响
    base_temp=20.0
    temp_amplitude=15.0#温度振幅
    temperature=base_temp+temp_amplitude*np.sin(2*np.pi*day_of_year/365-np.pi/2)+\
    np.random.normal(0,3,size=hours)#加入一些噪声
    #极端温度增加负荷
    temp_effect=1+0.02*np.abs(temperature-20)
    #添加随机波动
    noise=np.random.normal(0,10,hours)
    #最终负荷
    load=daily_pattern*weekly_pattern*seasonal_pattern*temp_effect+noise
    load=np.clip(load,50,300)#限制在50-300之间
    #创建DataFrame
    df=pd.DataFrame({
      'DateTime':dates,
      'Load':load,
      'Temperature':temperature,
      'Hour':hour_of_day,
      'DayOfWeek':day_of_week

    })
    return df
#生成数据
power_data=generate_power_data(2000)
print("电力负荷数据预览：")
print(power_data.head())
print(f"\n数据形状:{power_data.shape}")

# 可视化电力负荷数据
plt.figure(figsize=(15, 6))
plt.plot(power_data['DateTime'][:168], power_data['Load'][:168], label='Power Load', linewidth=1, color='orange')
plt.title('Simulated Power Load Data (First Week)')
plt.xlabel('DateTime')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
# 可视化负荷与温度的关系
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 负荷随时间变化
axes[0].scatter(power_data['Temperature'][:500], power_data['Load'][:500], alpha=0.5, color='red')
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Load (MW)')
axes[0].set_title('Load vs Temperature')
axes[0].grid(True)

# 不同小时的负荷分布
hourly_load = [power_data[power_data['Hour'] == h]['Load'].values for h in range(24)]
axes[1].boxplot(hourly_load)
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Load (MW)')
axes[1].set_title('Load Distribution by Hour')
axes[1].grid(True)
#plt.tight_layout()
#plt.show()

#数据预处理
def prepare_data(data,lookback_hours=24,test_ratio=0.2):
    """
    data:电力负荷数据的DataFrame
    lookback_hours:用于预测的历史时间小时数
    test_ratio:测试集比例
    """
    #特征选择
    features=["Load","Temperature","Hour","DayOfWeek"]
    dataset=data[features].values
    #数据归一化
    scaler=MinMaxScaler(feature_range=(0,1))#0和1代表归一化后的范围
    scaled_data=scaler.fit_transform(dataset)#对数据进行归一化

    #创建训练数据
    X,y=[],[]
    for i in range(lookback_hours,len(dataset)):
        X.append(scaled_data[i-lookback_hours:i])#过去lookback_hours小时的数据作为输入
        y.append(scaled_data[i,0])#当前小时的负荷作为输出,0代表负荷
    X,y=np.array(X),np.array(y)#转换为numpy数组，因为输入和输出都是数组

    #划分训练集和测试集
    split_idx = int(len(X) * (1 - test_ratio))
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]
    
    return train_X, train_y, test_X, test_y, scaler
#数据准备
loockback_hours = 24
train_X, train_y, test_X, test_y, scaler = prepare_data(power_data)
print(f"训练数据形状: X={train_X.shape}, y={train_y.shape}")
print(f"测试数据形状: X={test_X.shape}, y={test_y.shape}")
print(f"\n输入数据示例 (前5小时的数据):\n{train_X[0, :5, :]}\n")
print(f"对应的目标值 (第25小时的负荷): {train_y[0]:.6f}")

#定义LSTM模型
class PowerLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,dropout=0.2):
        super(PowerLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        

        #LSTM层
        self.lstm=nn.LSTM(
            input_size=input_size,#输入特征数
            hidden_size=hidden_size,#隐藏层特征数
            num_layers=num_layers,#LSTM层数
            batch_first=True,#输入和输出的维度顺序为(batch_size,sequence_length,feature_size)
            dropout=dropout,#dropout概率,dropout意思是在训练过程中，随机将一部分神经元的输出置零，防止过拟合
        )
        self.dropout=nn.Dropout(dropout)#dropout层
        #全连接层
        self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        #x的维度为(batch_size,sequence_length,feature_size)
        #初始化隐藏状态以及细胞状态
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        #通过LSTM层
        out,_=self.lstm(x,(h0,c0))
        #只是用最后一个时间步的输出
        out=self.dropout(out[:,-1,:])
        #通过全连接层
        out=self.fc(out)
        return out
    
    #创建模型实例
input_size=train_X.shape[2]#输入特征数
hidden_size=64
num_layers=2
output_size=1
model=PowerLSTM(input_size,hidden_size,num_layers,output_size)
print(model)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")

#模型训练
train_X_tensor=torch.FloatTensor(train_X)
train_y_tensor=torch.FloatTensor(train_y)
test_X_tensor=torch.FloatTensor(test_X)
test_y_tensor=torch.FloatTensor(test_y)
#创建数据加载器
batch_size=32#批次大小
train_dataset=torch.utils.data.TensorDataset(train_X_tensor,train_y_tensor)#创建数据集加载器
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)#创建数据加载器
#定义损失函数和优化器
criterion=nn.MSELoss()#均方误差损失
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)#Adam优化器

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将模型移到相应设备
model.to(device)
print(f"\n训练数据批次数量: {len(train_loader)}")
print(f"批次大小: {batch_size}")

#训练函数
def train_model(model,train_loader,criterion,optimizer,num_epochs=100):
    model.train()#设置模型为训练模式
    losses=[]#损失列表
    for epoch in range(num_epochs):
        epoch_loss=0
        for batch_X,batch_y in train_loader:
            #将数据转移到设备
            batch_X,batch_y=batch_X.to(device),batch_y.to(device)
            optimizer.zero_grad()#清零梯度
            # 前向传播
            outputs=model(batch_X)
            loss=criterion(outputs.squeeze(),batch_y)
            #反向传播
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()#更新参数
            epoch_loss+=loss.item()#.item()是把张量转换为标量，int，float，long等类型
            avg_loss=epoch_loss/len(train_loader)
            losses.append(avg_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses
    
#训练模型
print("\n开始训练...")
train_losses=train_model(model,train_loader,criterion,optimizer,num_epochs=100)
print("训练完成。")