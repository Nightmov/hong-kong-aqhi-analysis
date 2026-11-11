"""Daily AQHI co-occurrence rules: length-two Apriori, temporal holdout and checks.

No extra dependency, no changes to the original clustering results. Rule arrows
express same-day conditional frequencies, not time order or pollution transport.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'results/tables'
FIGURES = ROOT/'results/figures'
CHECKS = ROOT/'results/checks'
SEASONS = {12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
           6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
METRICS = ['transactions','antecedent_days','consequent_days','joint_days',
           'antecedent_support','consequent_support','support','confidence','lift','leverage']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_table(frame,name,index=False):
    frame.to_csv(TABLES/(name+'.csv'),index=index,encoding='utf-8-sig',float_format='%.12g')

def metrics(n,na,nb,nab):
    support=nab/n if n else np.nan
    confidence=nab/na if na else np.nan
    baseline=nb/n if n else np.nan
    return dict(zip(METRICS,[n,na,nb,nab,na/n if n else np.nan,baseline,support,
                            confidence,confidence/baseline if nb and n else np.nan,
                            support-(na/n)*(nb/n) if n else np.nan]))

def daily_data(cfg):
    inp=(ROOT/cfg['input_directory']).resolve()
    source=inp/'aqhi_long.csv'
    d=pd.read_csv(source,usecols=['date','hour','station','aqhi','is_estimated'])
    stations=sorted(set(d.station)-set(cfg['excluded_stations']))
    assert len(stations)==cfg['expected_station_count']
    timestamp=pd.to_datetime(d.date)+pd.to_timedelta(d.hour,unit='h')
    assert cfg['time_mapping']=='source_date_plus_hour'
    mask=timestamp.ge(cfg['main_start']) & timestamp.lt(cfg['main_end_exclusive']) & d.station.isin(stations)
    d=d.loc[mask].copy(); timestamp=timestamp.loc[mask]
    assert not pd.DataFrame({'station':d.station,'timestamp':timestamp}).duplicated().any()
    d['calendar_date']=timestamp.dt.normalize()
    d['valid']=(d.aqhi.notna() & (d.is_estimated.eq(0) if cfg['exclude_estimated'] else True)).astype(int)
    d['high']=(d.valid.eq(1) & d.aqhi.ge(cfg['high_risk_threshold'])).astype(int)
    dates=pd.date_range(cfg['main_start'],cfg['main_end_exclusive'],freq='D',inclusive='left')
    grid=pd.MultiIndex.from_product([dates,stations],names=['date','station'])
    counts=d.groupby(['calendar_date','station'])[['valid','high']].sum()
    counts.index.names=['date','station']
    daily=counts.reindex(grid,fill_value=0).reset_index()
    daily['expected_hours']=24
    daily['coverage']=daily.valid/24
    assert daily.valid.between(0,24).all() and (daily.high<=daily.valid).all()
    valid=daily.pivot(index='date',columns='station',values='valid').reindex(columns=stations)
    high=daily.pivot(index='date',columns='station',values='high').reindex(columns=stations)
    return daily,valid,high,source

def transactions(valid,high,min_valid,min_high):
    eligible=valid.ge(min_valid).all(axis=1)
    # Keep eligible all-zero days: they belong in every support denominator.
    return high.loc[eligible].ge(min_high).astype(bool)

def apriori_two(tx,min_support):
    """Apriori L1 -> join/prune -> L2, intentionally stopping at itemsets of size 2."""
    n=len(tx)
    if not n: raise ValueError('No eligible discovery transactions')
    singles={s:int(tx[s].sum()) for s in tx.columns}
    frequent={s for s,count in singles.items() if count/n>=min_support}
    rows=[{'size':1,'item_a':s,'item_b':'','count':singles[s],'support':singles[s]/n}
          for s in sorted(frequent)]
    for a,b in itertools.combinations(sorted(frequent),2):
        count=int((tx[a] & tx[b]).sum())
        if count/n>=min_support:
            rows.append({'size':2,'item_a':a,'item_b':b,'count':count,'support':count/n})
    return pd.DataFrame(rows,columns=['size','item_a','item_b','count','support'])

def all_pairs(tx):
    counts=tx.to_numpy(dtype=np.int64)
    totals=counts.sum(axis=0); joint=counts.T@counts
    return pd.DataFrame([{'antecedent':a,'consequent':b,**metrics(len(tx),int(totals[i]),int(totals[j]),int(joint[i,j]))}
                         for i,a in enumerate(tx.columns) for j,b in enumerate(tx.columns) if i!=j])

def passes(r,settings,validation=False):
    return (r.support.ge(settings['minimum_support']) & r.confidence.ge(settings['minimum_confidence'])
            & r.lift.ge(settings['minimum_lift'])
            & r.antecedent_days.ge(settings['validation_minimum_antecedent_days'] if validation else settings['minimum_antecedent_days'])
            & r.joint_days.ge(settings['validation_minimum_joint_days'] if validation else settings['minimum_joint_days']))

def interval_bootstrap(tx,candidates,start,end,repetitions,seed):
    """Calendar-month-stratified block bootstrap on the frozen candidate list."""
    output={}
    if candidates.empty: return output
    months=pd.period_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='M')
    names=tx.columns.tolist(); ids={s:i for i,s in enumerate(names)}
    a=np.array([ids[s] for s in candidates.antecedent]); b=np.array([ids[s] for s in candidates.consequent])
    blocks=[]
    keys=tx.index.to_period('M')
    for month in months:
        z=tx.loc[keys==month].to_numpy(dtype=np.int64)
        totals=z.sum(axis=0); joint=z.T@z
        blocks.append(np.column_stack([np.full(len(a),len(z)),totals[a],totals[b],joint[a,b]]))
    blocks=np.asarray(blocks)
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(repetitions):
        sampled=[]
        for m in range(1,13):
            pool=np.where(months.month==m)[0]
            sampled.extend(rng.choice(pool,len(pool),replace=True))
        n,na,nb,nab=blocks[sampled].sum(axis=0).T
        confidence=np.divide(nab,na,out=np.full(len(a),np.nan),where=na>0)
        lift=np.divide(nab*n,na*nb,out=np.full(len(a),np.nan),where=(na*nb)>0)
        estimates.append(np.column_stack([confidence,lift]))
    arr=np.asarray(estimates)
    for j in range(len(candidates)):
        for k,name in enumerate(['confidence','lift']):
            values=arr[:,j,k]; values=values[np.isfinite(values)]
            lo,hi=np.quantile(values,[.025,.975]) if len(values) else (np.nan,np.nan)
            output.setdefault(name+'_ci_low',[]).append(float(lo))
            output.setdefault(name+'_ci_high',[]).append(float(hi))
            output.setdefault(name+'_bootstrap_valid',[]).append(len(values))
    return output

def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); settings=cfg['association_rules']
    for p in [TABLES,FIGURES,CHECKS]: p.mkdir(parents=True,exist_ok=True)
    assert pd.Timestamp(cfg['main_start'])<pd.Timestamp(settings['validation_start'])<pd.Timestamp(cfg['main_end_exclusive'])
    print('Preparing daily station events...',flush=True)
    daily,valid,high,source=daily_data(cfg)
    tx=transactions(valid,high,settings['minimum_daily_valid_hours'],settings['minimum_daily_high_hours'])
    split=pd.Timestamp(settings['validation_start'])
    train=tx.loc[tx.index<split]; test=tx.loc[tx.index>=split]
    if train.empty or test.empty: raise ValueError('Discovery or validation has no eligible dates')
    daily['eligible_station_day']=daily.valid.ge(settings['minimum_daily_valid_hours'])
    daily['event']=np.where(daily.eligible_station_day,daily.high.ge(settings['minimum_daily_high_hours']).astype(float),np.nan)
    write_table(daily,'association_station_daily')
    write_table(tx.astype(int),'association_daily_transactions',True)
    coverage=pd.DataFrame({'expected_stations':valid.shape[1],'minimum_valid_hours':valid.min(axis=1),
                           'insufficient_stations':valid.lt(settings['minimum_daily_valid_hours']).sum(axis=1),
                           'eligible':valid.index.isin(tx.index),'valid_station_hours':valid.sum(axis=1)},index=valid.index)
    coverage['period']=np.where(coverage.index<split,'discovery','validation')
    coverage['season']=coverage.index.month.map(SEASONS)
    write_table(coverage,'association_date_coverage',True)
    cov=coverage.reset_index(); cov['year']=cov.date.dt.year
    summaries=[]
    for dimension in ['year','season']:
        g=cov.groupby(['period',dimension]).agg(expected_days=('eligible','size'),eligible_days=('eligible','sum')).reset_index()
        g['dimension']=dimension; g['value']=g[dimension].astype(str); g['retention']=g.eligible_days/g.expected_days
        summaries.append(g.drop(columns=dimension))
    write_table(pd.concat(summaries,ignore_index=True),'association_coverage_summary')
    frequent=apriori_two(train,settings['minimum_support'])
    write_table(frequent,'association_frequent_itemsets')
    tr=all_pairs(train); te=all_pairs(test); pooled=all_pairs(tx)
    keys=['antecedent','consequent']
    full=tr.rename(columns={k:'train_'+k for k in METRICS}).merge(
        te.rename(columns={k:'test_'+k for k in METRICS}),on=keys,validate='1:1').merge(
        pooled.rename(columns={k:'pooled_'+k for k in METRICS}),on=keys,validate='1:1')
    candidate_keys=set()
    for row in frequent.loc[frequent['size'].eq(2)].itertuples():
        candidate_keys.update([(row.item_a,row.item_b),(row.item_b,row.item_a)])
    selected=passes(tr,settings) & pd.Series([(a,b) in candidate_keys for a,b in zip(tr.antecedent,tr.consequent)],index=tr.index)
    candidates=full.loc[selected].copy().sort_values(['train_confidence','train_lift','train_support','antecedent','consequent'],
                                                    ascending=[False,False,False,True,True]).reset_index(drop=True)
    candidates.insert(0,'discovery_rank',np.arange(1,len(candidates)+1))
    test_metrics=candidates[[f'test_{k}' for k in METRICS]].rename(columns=lambda k:k[5:])
    candidates['validation_pass']=passes(test_metrics,settings,True)
    enough=test_metrics.antecedent_days.ge(settings['validation_minimum_antecedent_days']) & test_metrics.joint_days.ge(settings['validation_minimum_joint_days'])
    candidates['validation_status']=np.where(candidates.validation_pass,'meets_thresholds',np.where(enough,'weaker_than_thresholds','insufficient_events'))
    candidates['validation_positive_lift']=test_metrics.lift.gt(1)
    cluster_file=TABLES/'station_clusters.csv'
    if cluster_file.exists():
        clusters=pd.read_csv(cluster_file).set_index('station').cluster
        for part in ['antecedent','consequent']: candidates[part+'_cluster']=candidates[part].map(clusters)
        candidates['same_cluster']=candidates.antecedent_cluster.eq(candidates.consequent_cluster)
    print(f'Dates: discovery={len(train)}, validation={len(test)}; frozen candidates={len(candidates)}',flush=True)
    for label,z,start,end in [('train',train,cfg['main_start'],settings['validation_start']),
                              ('test',test,settings['validation_start'],cfg['main_end_exclusive'])]:
        intervals=interval_bootstrap(z,candidates,start,end,settings['bootstrap_repetitions'],cfg['random_seed']+(label=='test'))
        for name,values in intervals.items(): candidates[label+'_'+name]=values
    # Save all 210 directed pair statistics, not only successful rules.
    write_table(full,'association_all_pairs')
    write_table(candidates,'association_rules')
    seasonal=[]; sensitivity=[]
    scenarios={'primary':tx,'at_least_3_high_hours':transactions(valid,high,settings['minimum_daily_valid_hours'],3),
               'complete_24_hours':transactions(valid,high,24,settings['minimum_daily_high_hours']),
               'exclude_2023_08':tx.loc[tx.index.to_period('M')!=pd.Period('2023-08')]}
    for scenario,transactions_ in scenarios.items():
        for period,mask in [('discovery',transactions_.index<split),('validation',transactions_.index>=split)]:
            scores=all_pairs(transactions_.loc[mask])
            scores=candidates[keys].merge(scores,on=keys,validate='1:1')
            scores['scenario']=scenario; scores['period']=period
            sensitivity.append(scores)
    # Common-15 coverage excludes many dates. Check the frozen rules again on
    # each pair's own adequately observed dates; denominators then vary by pair.
    for period,start,end in [('discovery',pd.Timestamp(cfg['main_start']),split),
                             ('validation',split,pd.Timestamp(cfg['main_end_exclusive']))]:
        rows=[]
        for rule in candidates.itertuples():
            a,b=rule.antecedent,rule.consequent
            keep=valid[[a,b]].ge(settings['minimum_daily_valid_hours']).all(axis=1) & (valid.index>=start) & (valid.index<end)
            pair=high.loc[keep,[a,b]].ge(settings['minimum_daily_high_hours'])
            rows.append({'antecedent':a,'consequent':b,
                         **metrics(len(pair),int(pair[a].sum()),int(pair[b].sum()),int((pair[a]&pair[b]).sum())),
                         'scenario':'pairwise_available_days','period':period})
        sensitivity.append(pd.DataFrame(rows,columns=keys+METRICS+['scenario','period']))
    for period,z in [('discovery',train),('validation',test)]:
        for season in ['Spring','Summer','Autumn','Winter']:
            subset=z.loc[z.index.month.map(SEASONS)==season]
            scores=candidates[keys].merge(all_pairs(subset),on=keys,validate='1:1')
            scores['period']=period; scores['season']=season; seasonal.append(scores)
    sensitivity_table=pd.concat(sensitivity,ignore_index=True)
    write_table(sensitivity_table,'association_sensitivity')
    write_table(pd.concat(seasonal,ignore_index=True),'association_rule_seasonal')
    # Held-out lift heatmap; low-count pairs remain unknown rather than shown as zero.
    names=tx.columns.tolist()
    if cluster_file.exists(): names=sorted(names,key=lambda x:(clusters.get(x,np.inf),x))
    matrix=te.pivot(index='antecedent',columns='consequent',values='lift').reindex(index=names,columns=names)
    adequate=te.joint_days.ge(settings['validation_minimum_joint_days']) & te.antecedent_days.ge(settings['validation_minimum_antecedent_days']) & te.consequent_days.ge(settings['validation_minimum_antecedent_days'])
    available=te.assign(available=adequate).pivot(index='antecedent',columns='consequent',values='available').reindex(index=names,columns=names).eq(True)
    matrix=matrix.where(available)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(10,8)); cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#ECECEC')
    values=matrix.to_numpy(); finite=values[np.isfinite(values)]; vmax=max(2,float(finite.max())) if finite.size else 2
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=vmax))
    ax.set_xticks(range(len(names)),names,rotation=90); ax.set_yticks(range(len(names)),names)
    ax.set_title('Same-day high-risk association: validation 2021-2024',fontweight='bold',pad=16)
    fig.colorbar(im,ax=ax,label='Lift (1 = independence baseline)')
    fig.text(.5,.01,'Gray: diagonal or insufficient event counts. Same-day co-occurrence, not causation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1])
    for extension in ['png','svg']: fig.savefig(FIGURES/f'association_lift_heatmap.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    summary={'settings':settings,'method':'Apriori L1/L2 join-and-prune; single-station antecedent/consequent only',
             'high_risk_threshold':cfg['high_risk_threshold'],'stations':names,
             'expected_dates':len(valid),'eligible_dates':len(tx),'excluded_dates':len(valid)-len(tx),
             'discovery_dates':len(train),'validation_dates':len(test),
             'discovery_empty_transactions':int((~train.any(axis=1)).sum()),
             'validation_empty_transactions':int((~test.any(axis=1)).sum()),
             'frequent_singletons':int(frequent['size'].eq(1).sum()),'frequent_pairs':int(frequent['size'].eq(2).sum()),
             'candidate_rules':len(candidates),'validation_pass_rules':int(candidates.validation_pass.sum()),
             'validation_positive_lift_rules':int(candidates.validation_positive_lift.sum()),
             'pairwise_coverage_validation_pass_rules':int(passes(sensitivity_table.loc[
                 sensitivity_table.scenario.eq('pairwise_available_days') & sensitivity_table.period.eq('validation')],settings,True).sum()),
             'input_file':'aqhi_long.csv','input_sha256':digest(source),
             'core_input_manifest_matches':None,'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__},
             'caveats':['All metrics are conditional on common eligible dates.','Validation rules are frozen using discovery data only.',
                        'Pooled metrics are descriptive, not held-out evidence.','Bootstrap fixes the selected rules; no causal or propagation inference.',
                        'Cluster comparison is descriptive because both analyses use AQHI.']}
    if (TABLES/'input_manifest.csv').exists():
        manifest=pd.read_csv(TABLES/'input_manifest.csv').set_index('file')
        summary['core_input_manifest_matches']=summary['input_sha256']==manifest.loc['aqhi_long.csv','sha256']
        assert summary['core_input_manifest_matches']
    (CHECKS/'association_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_explanation(summary,candidates)
    print(json.dumps({k:summary[k] for k in ['eligible_dates','discovery_dates','validation_dates','candidate_rules','validation_pass_rules']},indent=2))

def write_explanation(s,rules):
    settings=s['settings']
    lines=['# 关联规则挖掘：监测站同日高风险共现\n',
      '## 做了什么\n',
      '使用2014—2024年固定15站，每天是一条事务。某站当天至少1个有效小时AQHI≥7，就记为该站发生高风险事件。剔除替代与缺失记录；15站当天均有至少20个有效小时才保留该日。合格但没有任何高风险事件的日期也保留，不将缺失当成0。规则箭头表示同日条件频率，不表示先后顺序、传播方向或因果。\n',
      f'完整日历有{s["expected_dates"]}天，保留{s["eligible_dates"]}天（{s["eligible_dates"]/s["expected_dates"]:.2%}），排除{s["excluded_dates"]}天。发现期2014—2020年保留{s["discovery_dates"]}天，验证期2021—2024年保留{s["validation_dates"]}天；其中无任何事件的日期分别为{s["discovery_empty_transactions"]}和{s["validation_empty_transactions"]}天。各年份/季节的筛选比例见association_coverage_summary.csv。共同覆盖要求排除了超过一半日期，因此结果仅代表保留日期，不能直接推广到全部日历日。\n',
      '## 方法与验证\n',
      '使用限定到两项集的Apriori：先找频繁单站，再将频繁单站连接成候选站对并筛选频繁二项集，最后生成单站→单站规则。仅依赖现有NumPy/pandas，没有增加安装依赖。\n',
      '支持度=两站共同事件天数/合格天数；置信度=共同事件天数/前项事件天数；提升度=置信度/后项本身的事件比例。提升度1为独立基线，A→B与B→A的置信度可能不同，但提升度相同。后项基准频率较低也可能使提升度较高，必须结合支持度、置信度与事件数解读。\n',
      '发现期筛选门槛：支持度≥1%、置信度≥60%、提升度≥1.2、前项至少50天、共同事件至少30天。候选完全由2014—2020年确定；2021—2024年不重新选规则。验证期沿用比例门槛，事件数门槛为前项至少20天、共同至少10天；“通过”仅表示达到这些分析门槛，不等于显著性检验。\n',
      f'共筛出{s["candidate_rules"]}条候选规则，验证期有{s["validation_pass_rules"]}条达到全部验证门槛，{s["validation_positive_lift_rules"]}条提升度大于1。这些是有向规则，同一站对的两个方向可各计一条，并非{s["candidate_rules"]}个独立站对。另分别在发现期与验证期各做{settings["bootstrap_repetitions"]}次月份分块重抽样，并按季节、至少3个高风险小时、完整24有效小时和排除2023年8月进行检查；所有检查沿用固定候选，不根据验证结果改阈值。\n',
      f'考虑共同15站覆盖筛选损失较多，额外对固定候选采用“每对站各自均有20有效小时”的日期重算；该敏感性检查中，验证期{s["pairwise_coverage_validation_pass_rules"]}条达到门槛。各规则分母不同，因此这项检查不用于重新挖掘或跨规则比较支持度。\n',
      '## 发现期排序前10条规则\n',
      '下表按发现期置信度、提升度、支持度排序，未按验证表现重新挑选。数值为真实计算结果；完整表含共同事件数、后项基准频率、区间和聚类标签。\n',
      '| 规则 | 发现期支持度 | 发现期置信度 | 发现期提升度 | 验证期共同天数 | 验证期置信度 | 验证期提升度 | 达到验证门槛 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rules.head(10).itertuples():
        lines.append(f'| {r.antecedent} → {r.consequent} | {r.train_support:.2%} | {r.train_confidence:.2%} | {r.train_lift:.2f} | {int(r.test_joint_days)} | {r.test_confidence:.2%} | {r.test_lift:.2f} | {"是" if r.validation_pass else "否"} |')
    if rules.empty: lines.append('\n没有规则达到发现期门槛，未为获得规则降低阈值。')
    lines+=['\n## 如何读图与结果文件\n',
      '![验证期站点同日高风险提升度](../figures/association_lift_heatmap.png)\n',
      '热图只使用2021—2024年验证数据，红色表示提升度高于1，蓝色低于1；灰色为对角线或事件数不足的站对，不表示关联为0。图中包含全部满足计数门槛的站对，未只展示筛选成功的规则。\n',
      '- association_rules.csv：发现期冻结规则，含验证结果及月份块区间。\n- association_all_pairs.csv：全部210个有向站对的发现/验证/合并统计。\n- association_daily_transactions.csv：合格日期的0/1事务矩阵。\n- association_station_daily.csv、association_date_coverage.csv：站日计数、未知标记和日期筛选。\n- association_frequent_itemsets.csv：发现期频繁一项集与二项集。\n- association_rule_seasonal.csv、association_sensitivity.csv：季节及事件/覆盖定义对照。\n',
      '连续高风险日可能来自同一过程，因此采用月份块评估不确定性。置信度不是预测准确率；高提升度也可能受季节、气象及筛选日期影响。与聚类的关系用于补充描述，不能作为完全独立验证。\n',
      '方法参考：[Apriori](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)；[关联规则指标](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)。']
    (TABLES/'association_rules_说明.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': main()
