#加载需要的库
import time
import math
import datetime
import numpy as np
import pandas as pd

import tushare as ts
import backtrader as bt
from sklearn.svm import SVR

from ast import literal_eval
import matplotlib.pyplot as plt
from backtrader.feeds import PandasData
from sklearn.preprocessing import RobustScaler

#实例化支持向量回归（SVR）模型
svr = SVR()

#获取已清洗好的全A股列表
stocklist_allA = pd.read_csv('Data/1-stocklist_allA.csv')
stocklist_allA = stocklist_allA['0'].tolist()

#获取已清洗好的全A股所有数据
df_all = pd.read_csv('Data/4-allA-all_data.csv')
del df_all['datetime.1']
df_all['datetime'] = pd.to_datetime(df_all['datetime'])

'''
获得某一股票的全部数据
输入：code--该股票对应的ts_code
输出：df_stock--该股票的全部数据，存为df
'''
def get_stock_data(code):
    df_stock = df_all[df_all['code']==code]
    df_stock = df_stock[['datetime','open','high','low','close','volume','openinterest','news','heat','turnover_rate_f','circ_mv','total_mv']]
    df_stock.index = df_stock.datetime
    df_stock = df_stock.sort_index()
    return df_stock

#修改原数据加载模块，以便能够加载更多自定义的因子数据
class Addmoredata(PandasData):
    lines = ('news','heat','turnover_rate_f','circ_mv','total_mv',)
    params = (('news',7),('heat',8),('turnover_rate_f',9),('circ_mv',10),('total_mv',11),)


#设置佣金和印花税率
class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.    
    '''
    params = (
        ('stamp_duty', 0.001), # 印花税率
        ('commission', 0.0005), # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        )
 
    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''
        print('self.p.commission',self.p.commission)
        if size > 0: # 买入，不考虑印花税
            return  size * price * self.p.commission * 100
        elif size < 0: # 卖出，考虑印花税
            return - size * price * (self.p.stamp_duty + self.p.commission*100)
        else:
            return 0 

#编写策略
class momentum_factor_strategy(bt.Strategy):
    #interval-换仓间隔，stocknum-持仓股票数
    params = (("interval",1),("stocknum",10),)

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('{}, {}'.format(dt.isoformat(), txt))

    def __init__(self):
        #已清洗过的每日可用股票列表
        self.df_byday = pd.read_csv('Data/3-allA_valid_stocklist_byday.csv')
        #交易天数，用于判断是否交易
        self.bar_num = 0
        #上次交易股票的列表
        self.last = []  

        # 记录以往订单，在调仓日要全部取消未成交的订单
        self.order_list = []

    def prenext(self):
        
        self.next()
        
    def next(self):
        #记录交易日期
        self.bar_num+=1
        print("当天日期:{}".format(str(self.datas[0].datetime.date(0))))
        #计算当日是否调仓
        if self.bar_num%self.p.interval==0 and self.bar_num > 3 * self.p.interval and self.datas[0].datetime.date(0) < datetime.date(2020,6,25):
            #得到当天的时间
            current_date=self.datas[0].datetime.date(0)
            print("交易日日期:{}".format(str(self.datas[0].datetime.date(0))))
            #获得上一调仓日时间
            prev_date=self.datas[0].datetime.date(-self.p.interval)
            #获取当日可行股票池
            stocklist = self.get_valid_list_day(current_date)
            #获取上一调仓日可行股票池
            stocklist_p = stocklist_p = self.get_valid_list_day(prev_date)

            #计算本期因子数据df_fac，并清洗
            df_fac = self.get_df_fac(stocklist=stocklist,prev=0)
            df_fac = df_fac.dropna(axis=0,how='any')

            #计算上期因子数据df_faxc_p，并清洗
            df_fac_p = self.get_df_fac(stocklist=stocklist_p,prev=1)
            df_fac_p = df_fac_p.dropna(axis=0,how='any')

            #本期因子排列命名
            df_fac.columns = ['code','news','heat','momentum_value','turnover_rate_f','circ_mv','total_mv']
            df_fac.index = df_fac.code.values

            #上期因子排列命名
            df_fac_p.columns = ['code','news','heat','momentum_value','turnover_rate_f','circ_mv','total_mv']
            df_fac_p.index = df_fac_p.code.values

            #舍弃X_p和Y中不同的index（股票代码）
            #先去除X_p比Y多的index
            diffIndex = df_fac_p.index.difference(df_fac.index)
            #删除整行
            df_fac_p = df_fac_p.drop(diffIndex,errors='ignore')
            df_fac = df_fac.drop(diffIndex,errors='ignore')

            # 然后去除Y比X_p多的index
            diffIndex = df_fac.index.difference(df_fac_p.index)
            df_fac_p = df_fac_p.drop(diffIndex,errors='ignore')
            df_fac = df_fac.drop(diffIndex,errors='ignore')

            #X_p是上一期的因子值，X是本期因子值，Y是回归目标
            X_p = df_fac_p[['news','heat','momentum_value','turnover_rate_f','circ_mv']]
            X = df_fac[['news','heat','momentum_value','turnover_rate_f','circ_mv']]
            Y = df_fac[['momentum_value']]

            #将因子值与Y值均进行标准化
            rbX = RobustScaler()
            X_p = rbX.fit_transform(X_p)

            rbY = RobustScaler()
            Y = rbY.fit_transform(Y)
            
            #用上期因子值与本期回报率进行训练
            svr.fit(X_p,np.ravel(Y)) 
            
            #用本期因子值预测下期回报率
            svr_pred = svr.predict(rbX.transform(X))
            
            a = rbY.inverse_transform(svr_pred.reshape(-1, 1))
            df_fac['pred'] = a
            
            #按照预测得到的下期收益进行排序
            df_fac.sort_values(by="pred" , inplace=True, ascending=False)
            #取预测收益>0且排序靠前的stocknum只股票做多
            df_fac_pos = df_fac[df_fac['pred']>0]
            sort_list_pos = df_fac['code'].tolist()
            long_list=sort_list_pos[:self.p.stocknum]
            
            #取预测收益<0且排序靠后的stocknum只股票做空
            df_fac_neg = df_fac[df_fac['pred']<0]
            sort_list_neg = df_fac_neg['code'].tolist()
            short_list=sort_list_neg[-self.p.stocknum:]            

            #取消以往所下订单（已成交的不会起作用）
            for o in self.order_list:
                self.cancel(o)
            #重置订单列表
            self.order_list = []  
            
            #若上期交易股票未出现在本期交易列表中，则平仓
            for i in self.last:
                if i not in long_list and i not in short_list:
                    d = self.getdatabyname(i)
                    print('sell 平仓', d._name, self.getposition(d).size)
                    o = self.close(data=d)
                    self.order_list.append(o) # 记录订单
            
            self.log('当前总市值 %.2f' %(self.broker.getvalue()))
            
            #获取当前账户价值
            total_value = self.broker.getvalue()
            
            #对long_list中股票做多
            if len(long_list):
        
                #每只股票买入资金百分比，预留5%的资金以应付佣金和计算误差
                buypercentage = (1-0.05)/2/len(long_list)
    
                #得到目标市值
                targetvalue = buypercentage * total_value
                
                #依次买入
                for d in long_list:
                    data = self.getdatabyname(d)
                    #按次日开盘价计算下单量，下单量是100的整数倍
                    size = int(abs(targetvalue / data.open[1] // 100 * 100))
                    o = self.order_target_size(data=d, target=size)
                    #记录订单             
                    self.order_list.append(o) 

            #对short_list中股票做空                    
            if len(short_list):
                
                #每只股票做空资金百分比，预留5%的资金以应付佣金和计算误差
                buypercentage = (1-0.05)/2/len(short_list)
    
                #得到目标市值
                targetvalue = buypercentage * total_value
                #依次卖空
                for d in short_list:
                    data = self.getdatabyname(d)
                    #按次日开盘价计算下单量，下单量是100的整数倍
                    size = int(abs(targetvalue / data.open[1] // 100 * 100))
                    o = self.order_target_size(data=d, target=-size)
                    #记录订单             
                    self.order_list.append(o) 
            
            #跟踪上次交易的标的
            self.last = list(set(long_list).union(set(short_list)))  
            
    #交易日志    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm,
                     order.executed.size,
                     order.data._name))
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s' %
                        (order.executed.price,
                         order.executed.value,
                         order.executed.comm,
                         order.executed.size,
                         order.data._name))
                

    def notify_trade(self, trade):
        if trade.isclosed:
            
            self.log('TRADE PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

    
    # 求个股某周期因子平均值，prev为是否为前一周期（0：否，1：是）
    def get_df_fac(self,stocklist,prev=0):
        #初始化若干个list，用于计算因子值
        #N-个股情绪因子，H-个股热度因子，TR_F-流通股换手率，CM-流通市值，TM-总市值（未使用）
        N,H,TR_F,CM,TM = [],[],[],[],[]
        #新建df_fac用于存储计算得到的因子值
        df_fac = pd.DataFrame(columns=['code','news','heat','momentum_value','turnover_rate_f','circ_mv','total_mv']) 
        for stock in stocklist:
            data=self.getdatabyname(stock)
            
            #获取当期因子值的平均数
            for i in range(self.p.interval):
                if data.news[(-i-prev*self.p.interval)] != 0:
                    N.append(data.news[(-i-prev*self.p.interval)])

                if data.heat[(-i-prev*self.p.interval)] != 0:
                    H.append(data.heat[(-i-prev*self.p.interval)])
                TR_F.append(data.turnover_rate_f[(-i-prev*self.p.interval)])
                CM.append(data.circ_mv[(-i-prev*self.p.interval)])
                TM.append(data.total_mv[(-i-prev*self.p.interval)])
            
            #若当期无舆情因子数据，则不返回
            if N and H:
                #缺失数据处理--若后续数据缺失，跳过该股票
                try:
                    data.open[1]
                except IndexError:
                    print('%s后续数据缺失，跳过该股票'%(stock))
                else:
                    if data.close[0] != 0 and data.open[1] !=0:
                        #舆情因子取最近一天的值与当期平均值之和（加大最近一天的权重）
                        N = np.mean(N) + data.news[(-prev*self.p.interval)]
                        H = np.mean(H) + data.heat[(-prev*self.p.interval)]
                    
                    try:
                        data.open[1+self.p.interval-prev*self.p.interval]
                    except IndexError:
                        print('%s后续数据缺失，跳过该股票'%(stock))
                    else:
    
                        #计算当期动量
                        sell_ = data.open[1+self.p.interval-prev*self.p.interval]
                        buy_ = data.open[1-prev*self.p.interval]
                        stock_momentum = sell_ / buy_ - 1
                        #其余因子取当期平均值
                        TR_F = np.mean(TR_F)
                        CM = np.mean(CM)
                        TM = np.mean(TM)
    
                        new = pd.DataFrame({'code':stock,'news':N,'heat':H,'momentum_value':stock_momentum,
                                            'turnover_rate_f':TR_F,'circ_mv':CM,'total_mv':TM},index=[1]) 
                        df_fac = df_fac.append(new,ignore_index=True)
            N,H,TR_F,CM,TM = [],[],[],[],[]
        return df_fac
    
    #获取当日可行股票池
    def get_valid_list_day(self,current_date):
        self.df_byday['Date'] = pd.to_datetime(self.df_byday['Date'])
        current_date = datetime.datetime.strptime(str(current_date),'%Y-%m-%d')
        df_day = self.df_byday[self.df_byday['Date']==current_date]
        stocklist = literal_eval(df_day['stocklist'].tolist()[0])
        return stocklist


##########################
# 主程序开始
##########################


begin_time=time.time()
cerebro = bt.Cerebro(stdstats=False)
#cerebro.broker = bt.brokers.BackBroker(shortcash=True)  # 0.5%
#考虑印花税和佣金印花税为单边千分之一，佣金设为万五
comminfo=stampDutyCommissionScheme(stamp_duty=0.001,commission=0.0005)
cerebro.broker.addcommissioninfo(comminfo)

for s in stocklist_allA:
    feed = Addmoredata(dataname = get_stock_data(s),plot=False,
                       fromdate=datetime.datetime(2017,1,3),todate=datetime.datetime(2020,6,30))
    cerebro.adddata(feed, name = s)


cerebro.broker.setcash(1000000.0)
#防止下单时现金不够被拒绝。只在执行时检查现金够不够。
cerebro.broker.set_checksubmit(False)
# 添加相应的费用，杠杆率
# 获取策略运行的指标
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

startcash=1000000.0
cerebro.addstrategy(momentum_factor_strategy) 
cerebro.addobserver(bt.observers.Value)

#添加Analyzer
cerebro.addanalyzer(
    bt.analyzers.SharpeRatio,
    riskfreerate=0.01,
    annualize=True,
    _name='sharp_ratio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

thestrats = cerebro.run()
thestrat = thestrats[0]
# 输出分析器结果字典
print('Sharpe Ratio:', thestrat.analyzers.sharp_ratio.get_analysis())
print('DrawDown:', thestrat.analyzers.drawdown.get_analysis())

# 进一步从字典中取出需要的值
print('Sharpe Ratio:',
      thestrat.analyzers.sharp_ratio.get_analysis()['sharperatio'])
print('Max DrawDown:',
      thestrat.analyzers.drawdown.get_analysis()['max']['drawdown'])

# 打印各个分析器内容
for a in thestrat.analyzers:
    a.print()
cerebro.plot() 
#获取回测结束后的总资金
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash
#打印结果
print(f'总资金: {round(portvalue,2)}')
print(f'净收益: {round(pnl,2)}')
end_time=time.time()
print("一共使用时间为:{}".format(end_time-begin_time))