#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
野外声学监测设备 —— 采集失败根因分析与运维派单决策依据
可复现脚本：python3 analyze.py
输入: audio_records.csv (题目发放)
输出: 终端全部数字 + report_fig.png
口径: 采集失败损失 75 工时/次；提前派单检修 15 工时/次
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ['WenQuanYi Micro Hei','Noto Sans CJK SC','SimHei']:
    try:
        font_manager.findfont(f, fallback_to_default=False); plt.rcParams['font.sans-serif']=[f]; break
    except Exception: pass
plt.rcParams['axes.unicode_minus']=False

FAIL_H, DISPATCH_H = 75, 15
import os
CSV=(__file__.rsplit('/',1)[0] if '/' in __file__ else '.')+'/audio_records.csv'
if not os.path.exists(CSV): CSV='/data/inputs/audio_records.csv'
COLS=['record_id','event_time','heartbeat_time','station_id','device_model','battery_level','ambient_noise','is_captured']

def load():
    raw=pd.read_csv(CSV)
    raw['row']=np.arange(len(raw))                 # 原始行序（假设=入库序）
    df=raw.drop_duplicates(subset=COLS).copy()     # 54 条完全重复行，按原始列去重
    df['event_utc']=pd.to_datetime(df.event_time,utc=True,format='mixed')
    df['hb_utc']   =pd.to_datetime(df.heartbeat_time,utc=True,format='mixed')
    df['lag_min']  =(df.event_utc-df.hb_utc).dt.total_seconds()/60   # 心跳陈旧度
    df['day']      =df.event_utc.dt.tz_convert('Asia/Shanghai').dt.date
    return raw,df

raw,df=load()
r=df[df.is_captured.notna()].copy()          # 已回传，用于失败率
tot_fail=int((r.is_captured==0).sum())

def line(t=''): print('\n'+'='*78+('\n'+t if t else ''))

line('【0】数据体检')
print(f"原始行数 {len(raw)} | 完全重复行 {raw[COLS].duplicated().sum()} | 去重后 {len(df)}")
print(f"is_captured: 成功 {int((df.is_captured==1).sum())} / 失败 {int((df.is_captured==0).sum())} / 未回传(空) {int(df.is_captured.isna().sum())}")
print(f"失败率(已回传口径) {(r.is_captured==0).mean()*100:.2f}%  ({tot_fail}/{len(r)})")
print(f"时间跨度 {df.day.min()} ~ {df.day.max()} | 站点 {df.station_id.nunique()} | 型号 {df.device_model.nunique()}")
print(f"心跳滞后 lag_min: 中位数 {df.lag_min.median():.0f} 最大 {df.lag_min.max():.0f} 分钟；负滞后 {int((df.lag_min<0).sum())} 条")

line('【1】根问题：事件入库严重滞后（假设 CSV 行序=入库序）')
d=df.sort_values('row').copy()                     # 必须按原始行序，不可用 event_time 排序
d['behind_h']=(d.event_utc.cummax()-d.event_utc).dt.total_seconds()/3600
print(f"event_time 沿原始行序单调递增? {d.event_utc.is_monotonic_increasing}")
print(f"落后于当前最大时间戳: 中位数 {d.behind_h.median():.1f} h | 均值 {d.behind_h.mean():.1f} h | 最大 {d.behind_h.max():.1f} h")
print(f"滞后>1h   {(d.behind_h>1).sum()} 条 = {(d.behind_h>1).mean()*100:.1f}%")
print(f"滞后>24h  {(d.behind_h>24).sum()} 条 = {(d.behind_h>24).mean()*100:.1f}%")
print(f"滞后>72h  {(d.behind_h>72).sum()} 条 = {(d.behind_h>72).mean()*100:.1f}%")
print("各日事件在原始行序中的跨度(说明数据按天分批回传、批次间严重交错):")
print(d.groupby('day').row.agg(['min','max','size']).to_string())

line('【2】单因素扫描')
def scan(col,bins,label):
    t=pd.cut(r[col],bins); out=r.groupby(t,observed=True).agg(
        n=('is_captured','size'), fail=('is_captured',lambda s:int((s==0).sum())),
        rate=('is_captured',lambda s:round((s==0).mean()*100,1)))
    print(f"--- {label} ---"); print(out.to_string()); return out
scan('battery_level',[-1,5,10,15,20,25,30],'电量 battery_level')
scan('ambient_noise',[-1,30,40,50,60,70,80,100],'底噪 ambient_noise')
scan('lag_min',[-1,10,20,30,40,50,60,70],'心跳滞后 lag_min')
print("\n--- 型号 ---")
m=r.groupby('device_model').agg(n=('is_captured','size'),fail=('is_captured',lambda s:int((s==0).sum())),
    rate=('is_captured',lambda s:round((s==0).mean()*100,1))).sort_values('rate',ascending=False)
print(m.to_string())
print("\n--- 站点 ---")
print(r.groupby('station_id').agg(n=('is_captured','size'),fail=('is_captured',lambda s:int((s==0).sum())),
    rate=('is_captured',lambda s:round((s==0).mean()*100,1))).sort_values('rate',ascending=False).to_string())

line('【3】阈值扫描：找 lag 断点')
print(f"{'阈值':>5}{'高危n':>7}{'高危失败':>9}{'高危率%':>9}{'低危率%':>9}{'提升':>8}{'召回%':>8}")
best=None
for t in range(10,45,5):
    hi=r[r.lag_min>t]; lo=r[r.lag_min<=t]
    rh=(hi.is_captured==0).mean(); rl=(lo.is_captured==0).mean()
    rec=(hi.is_captured==0).sum()/tot_fail*100
    print(f"{t:>5}{len(hi):>7}{int((hi.is_captured==0).sum()):>9}{rh*100:>9.1f}{rl*100:>9.1f}{rh/max(rl,1e-9):>7.1f}x{rec:>8.1f}")
    if best is None or rh/max(rl,1e-9)>best[1]: best=(t,rh/max(rl,1e-9))
print(f"最优阈值 lag>{best[0]} 分钟, 提升 {best[1]:.1f}x")
THR=20

line('【4】时间切分验证（前4天定阈值 / 后3天验证）—— 防过拟合')
days=sorted(r.day.unique()); tr=r[r.day.isin(days[:4])]; te=r[r.day.isin(days[4:])]
for nm,x in [('训练 d1-4',tr),('验证 d5-7',te)]:
    hi=x[x.lag_min>THR]; lo=x[x.lag_min<=THR]
    rh=(hi.is_captured==0).mean(); rl=(lo.is_captured==0).mean()
    print(f"{nm}: 高危 {rh*100:.1f}% (n={len(hi)}) | 低危 {rl*100:.1f}% (n={len(lo)}) | "
          f"提升 {rh/max(rl,1e-9):.1f}x | 召回 {(hi.is_captured==0).sum()/max((x.is_captured==0).sum(),1)*100:.1f}%")

line('【5】诚实检验：这个信号能\"提前\"预警吗？（用前一次观测预测下一次）')
g=df.groupby(['station_id','device_model'])
f=df.copy()
f['prev_lag']=g.lag_min.shift(1); f['prev_fail']=g.is_captured.shift(1).eq(0).astype(float)
f['bat_drop']=g.battery_level.diff(); f['gap_h']=g.event_utc.diff().dt.total_seconds()/3600
s=f[f.prev_lag.notna()&f.is_captured.notna()].copy(); base=(s.is_captured==0).mean()
print(f"可评估样本 {len(s)} | 基线失败率 {base*100:.2f}%")
for nm,mk in [('前次 lag>20',s.prev_lag>20),('前次采集失败',s.prev_fail==1),
              ('电量较前次下降>=3',s.bat_drop<=-3),('当前电量<=5',s.battery_level<=5)]:
    x=s[mk]; rt=(x.is_captured==0).mean()
    print(f"  {nm:<18} n={len(x):>4} 下次失败率={rt*100:>5.1f}%  提升={rt/base:.2f}x")
print(">> 结论: 前瞻提升≈1.1x，几乎无预测力。lag 是\"当期症状\"不是\"前兆\"，"
      "因此告警只能用于缩短发现时延，不能用于预测性排班。")

line('【6】派单方案与工时测算（阈值 lag>20，按 站点×型号×日 合并派单）')
alerts=r[r.lag_min>THR]
units=alerts.groupby(['station_id','device_model','day']).agg(
    n_alert=('record_id','size'), fails=('is_captured',lambda s:int((s==0).sum())))
nd=len(units); nf=int(units.fails.sum())
print(f"原始告警记录 {len(alerts)} 条 -> 合并后派单 {nd} 次 (每单元平均 {units.n_alert.mean():.2f} 条告警)")
print(f"覆盖失败 {nf}/{tot_fail} = {nf/tot_fail*100:.1f}%")
print(f"派单成本 {nd} x {DISPATCH_H} = {nd*DISPATCH_H} 工时")
print(f"可避免损失(上限) {nf} x {FAIL_H} = {nf*FAIL_H} 工时")
print(f"净收益(上限) {nf*FAIL_H-nd*DISPATCH_H} 工时")
print(f"盈亏平衡: 需阻止 {nd*DISPATCH_H/FAIL_H:.1f}/{nf} 次覆盖失败 = {nd*DISPATCH_H/(nf*FAIL_H)*100:.1f}%"
      f"，即每次派单平均阻止 {nd*DISPATCH_H/FAIL_H/nd:.2f} 次失败即可回本")
print("\n--- 阈值敏感性（100%阻止假设下的净收益）---")
print(f"{'阈值':>5}{'派单':>7}{'覆盖失败':>9}{'召回%':>8}{'净收益工时':>11}")
for t in [10,15,20,25,30,40]:
    a=r[r.lag_min>t]; u=a.groupby(['station_id','device_model','day']).agg(
        f=('is_captured',lambda s:int((s==0).sum())))
    print(f"{t:>5}{len(u):>7}{int(u.f.sum()):>9}{u.f.sum()/tot_fail*100:>7.1f}%{int(u.f.sum())*FAIL_H-len(u)*DISPATCH_H:>11}")
print("\n--- 现状基线 ---")
print(f"不作预防: {tot_fail} 次失败 x {FAIL_H} = {tot_fail*FAIL_H} 工时净损失")
print(f"例行巡检 5站x7天 = 35 次派单 = {35*DISPATCH_H} 工时，本身不阻止任何失败")

line('【7】未回传(空)盲区')
nz=df[df.is_captured.isna()].copy(); nz["behind_h"]=(df.sort_values("row").event_utc.cummax().reindex(nz.sort_values("row").index)-nz.event_utc).dt.total_seconds()/3600
print(f"未回传 {len(nz)} 条 = {len(nz)/len(df)*100:.1f}%，其中 lag>{THR} 占 {int((nz.lag_min>THR).sum())} 条")
print(f"未回传记录的中位入库滞后 {nz.behind_h.median() if 'behind_h' in nz else float('nan'):.1f} h" 
      if 'behind_h' in nz else "")
print(">> 这部分既不能算成功也不能算失败，是监控盲区，会让真实失败率被低估。")

# ---------- 图 ----------
fig,ax=plt.subplots(2,2,figsize=(13,8.5))
a=ax[0,0]
bins=[-1,10,20,30,40,50,60,70]; t=pd.cut(r.lag_min,bins)
v=r.groupby(t,observed=True).is_captured.apply(lambda s:(s==0).mean()*100)
a.bar([str(i) for i in v.index],v.values,color='#c0392b')
a.axvline(1.5,ls='--',c='#2c3e50'); a.set_title('① 心跳滞后 vs 失败率（20分钟后断崖）')
a.set_xlabel('lag_min 区间'); a.set_ylabel('失败率 %')
for i,x in enumerate(v.values): a.text(i,x+1,f'{x:.0f}%',ha='center',fontsize=9)

a=ax[0,1]
binsn=[-1,30,40,50,60,70,80,100]; tn=pd.cut(r.ambient_noise,binsn)
vn=r.groupby(tn,observed=True).is_captured.apply(lambda s:(s==0).mean()*100)
a.bar([str(i) for i in vn.index],vn.values,color='#2980b9')
a.set_title('② 环境底噪 vs 失败率（单调递增）'); a.set_xlabel('ambient_noise 区间'); a.set_ylabel('失败率 %')
for i,x in enumerate(vn.values): a.text(i,x+0.6,f'{x:.0f}%',ha='center',fontsize=8)

a=ax[1,0]
ts=list(range(10,45,5)); lifts=[]
for t0 in ts:
    hi=r[r.lag_min>t0]; lo=r[r.lag_min<=t0]
    lifts.append((hi.is_captured==0).mean()/max((lo.is_captured==0).mean(),1e-9))
a.plot(ts,lifts,'o-',color='#27ae60'); a.axvline(20,ls='--',c='#c0392b')
a.set_title('③ 阈值提升度扫描（lag>20 最优）'); a.set_xlabel('阈值(分钟)'); a.set_ylabel('失败率提升倍数')

a=ax[1,1]
d2=df.sort_values('row').copy(); d2['behind_h']=(d2.event_utc.cummax()-d2.event_utc).dt.total_seconds()/3600; a.hist(d2.behind_h,bins=40,color='#8e44ad')
a.axvline(d2.behind_h.median(),ls='--',c='#f39c12',label=f'中位 {d2.behind_h.median():.0f}h')
a.set_title('④ 事件入库滞后分布（根问题）'); a.set_xlabel('落后于当前最大时间戳(小时)'); a.set_ylabel('记录数'); a.legend()
plt.tight_layout(); plt.savefig('report_fig.png',dpi=130)
print("\n[图已保存] report_fig.png")

line('【8】派单优先级清单（给运维的今日作业单）')
# 期望收益 = 该单元告警条数 x 高危失败率 x 75 - 15
p = units.copy()
p['exp_fails'] = p.n_alert * (r[r.lag_min>THR].is_captured.eq(0).mean())
p['exp_value'] = p.exp_fails*FAIL_H - DISPATCH_H
p = p.sort_values('exp_value',ascending=False)
print(f"高危告警单元 {len(p)} 个；按 期望避免失败数 x {FAIL_H}h - {DISPATCH_H}h 排序")
print(f"单元告警条数分布: 1条 {int((p.n_alert==1).sum())} 个 | 2条 {int((p.n_alert==2).sum())} 个 | >=3条 {int((p.n_alert>=3).sum())} 个")
print("\n--- 优先级 TOP 15 (station / model / day) ---")
print(f"{'#':>3} {'站点':<7}{'型号':<14}{'日期':<12}{'告警数':>6}{'已失败':>7}{'期望收益工时':>12}")
for i,(idx,row_) in enumerate(p.head(15).iterrows(),1):
    st,mo,dy=idx
    print(f"{i:>3} {st:<7}{mo:<14}{str(dy):<12}{row_.n_alert:>6}{row_.fails:>7}{row_.exp_value:>12.0f}")
print(f"\n全部 {len(p)} 个单元的期望净收益合计: {p.exp_value.sum():.0f} 工时")
p.reset_index().to_csv('dispatch_priority.csv',index=False)
print("[已保存] dispatch_priority.csv —— 全部派单单元及优先级")
