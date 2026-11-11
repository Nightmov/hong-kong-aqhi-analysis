"""Independently reconstruct events and verify discovery, holdout and sensitivity."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from association_rules import apriori_two

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'results/tables'

def assert_close(actual,expected):
    assert np.allclose(np.asarray(actual,dtype=float),np.asarray(expected,dtype=float),rtol=1e-10,atol=1e-10,equal_nan=True)

def manual_metrics(tx,a,b):
    n=len(tx); na=int(tx[a].sum()); nb=int(tx[b].sum()); nab=int((tx[a]&tx[b]).sum())
    return {'transactions':n,'antecedent_days':na,'consequent_days':nb,'joint_days':nab,
            'antecedent_support':na/n if n else np.nan,'consequent_support':nb/n if n else np.nan,
            'support':nab/n if n else np.nan,'confidence':nab/na if na else np.nan,
            'lift':nab*n/(na*nb) if na and nb else np.nan,
            'leverage':nab/n-na*nb/n**2 if n else np.nan}

def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8')); s=cfg['association_rules']
    source=(ROOT/cfg['input_directory']/'aqhi_long.csv').resolve()
    d=pd.read_csv(source,usecols=['date','hour','station','aqhi','is_estimated'])
    d['time']=pd.to_datetime(d.date)+pd.to_timedelta(d.hour,unit='h')
    names=sorted(set(d.station)-set(cfg['excluded_stations']))
    d=d.loc[d.time.ge(cfg['main_start'])&d.time.lt(cfg['main_end_exclusive'])&d.station.isin(names)]
    d=d.loc[d.aqhi.notna() & (d.is_estimated.eq(0) if cfg['exclude_estimated'] else True)].copy()
    d['day']=d.time.dt.normalize(); d['high']=d.aqhi.ge(cfg['high_risk_threshold'])
    days=pd.date_range(cfg['main_start'],cfg['main_end_exclusive'],freq='D',inclusive='left')
    index=pd.MultiIndex.from_product([days,names],names=['day','station'])
    # Count directly from valid hourly rows rather than reusing the analysis builder.
    counts=d.groupby(['day','station']).agg(valid=('aqhi','size'),high=('high','sum')).reindex(index,fill_value=0)
    valid=counts.valid.unstack('station'); high=counts.high.unstack('station')
    def tx(min_hours,min_events):
        keep=valid.ge(min_hours).all(axis=1)
        return high.loc[keep].ge(min_events)
    primary=tx(s['minimum_daily_valid_hours'],s['minimum_daily_high_hours'])
    saved=pd.read_csv(T/'association_daily_transactions.csv',index_col='date',parse_dates=True).astype(bool)
    assert saved.index.equals(primary.index)
    assert saved.columns.tolist()==primary.columns.tolist()
    assert np.array_equal(saved.to_numpy(),primary.to_numpy())
    daily=pd.read_csv(T/'association_station_daily.csv',parse_dates=['date']).set_index(['date','station'])
    assert np.array_equal(daily[['valid','high']].to_numpy(),counts[['valid','high']].to_numpy())
    unknown=daily.valid.lt(s['minimum_daily_valid_hours'])
    assert daily.loc[unknown,'event'].isna().all()
    assert_close(daily.loc[~unknown,'event'],daily.loc[~unknown,'high'].ge(s['minimum_daily_high_hours']).astype(int))
    date_cov=pd.read_csv(T/'association_date_coverage.csv',index_col='date',parse_dates=True)
    assert len(date_cov)==len(days) and np.array_equal(date_cov.eligible.to_numpy(),days.isin(primary.index))
    split=pd.Timestamp(s['validation_start']); train=primary.loc[primary.index<split]; test=primary.loc[primary.index>=split]
    assert set(train.index).isdisjoint(test.index)
    pairs=pd.read_csv(T/'association_all_pairs.csv'); rules=pd.read_csv(T/'association_rules.csv')
    assert len(pairs)==len(names)*(len(names)-1)==210
    expected_candidates=set(); errors=[]
    for row in pairs.itertuples():
        for prefix,z in [('train',train),('test',test),('pooled',primary)]:
            direct=manual_metrics(z,row.antecedent,row.consequent)
            for key,value in direct.items():
                actual=getattr(row,prefix+'_'+key)
                assert_close(actual,value)
                if np.isfinite(value): errors.append(abs(actual-value))
        # Freeze membership using discovery numbers only.
        if (row.train_support>=s['minimum_support'] and row.train_confidence>=s['minimum_confidence']
                and row.train_lift>=s['minimum_lift'] and row.train_antecedent_days>=s['minimum_antecedent_days']
                and row.train_joint_days>=s['minimum_joint_days']):
            expected_candidates.add((row.antecedent,row.consequent))
    assert set(zip(rules.antecedent,rules.consequent))==expected_candidates
    for row in rules.itertuples():
        direct=manual_metrics(test,row.antecedent,row.consequent)
        passed=(direct['support']>=s['minimum_support'] and direct['confidence']>=s['minimum_confidence']
                and direct['lift']>=s['minimum_lift'] and direct['antecedent_days']>=s['validation_minimum_antecedent_days']
                and direct['joint_days']>=s['validation_minimum_joint_days'])
        assert bool(row.validation_pass)==passed
        for prefix in ['train','test']:
            for metric in ['confidence','lift']:
                lo=getattr(row,f'{prefix}_{metric}_ci_low'); hi=getattr(row,f'{prefix}_{metric}_ci_high')
                count=getattr(row,f'{prefix}_{metric}_bootstrap_valid')
                assert 0<=count<=s['bootstrap_repetitions']
                if count:
                    assert 0<=lo<=hi
                    if metric=='confidence': assert hi<=1
                else:
                    assert pd.isna(lo) and pd.isna(hi)
    frequent=pd.read_csv(T/'association_frequent_itemsets.csv').fillna({'item_b':''})
    expected_itemsets={}
    for k in [1,2]:
        for items in itertools.combinations(names,k):
            count=int(train[list(items)].all(axis=1).sum())
            if count/len(train)>=s['minimum_support']: expected_itemsets[items]=count
    actual_itemsets={tuple([r.item_a] if r.size==1 else [r.item_a,r.item_b]):int(r.count) for r in frequent.itertuples()}
    assert actual_itemsets==expected_itemsets
    for row in frequent.itertuples(): assert_close(row.support,row.count/len(train))
    # Known tiny fixture catches omission of empty transactions and wrong denominators.
    fixture=pd.DataFrame({'A':[1,1,0,0],'B':[1,0,1,0],'C':[0,0,0,0]},dtype=bool)
    tiny=apriori_two(fixture,.25)
    assert set(zip(tiny.item_a,tiny.item_b))=={('A',''),('B',''),('A','B')}
    result=manual_metrics(fixture,'A','B')
    assert result['support']==.25 and result['confidence']==.5 and result['lift']==1
    sensitivity=pd.read_csv(T/'association_sensitivity.csv')
    scenarios={'primary':primary,'at_least_3_high_hours':tx(s['minimum_daily_valid_hours'],3),
               'complete_24_hours':tx(24,s['minimum_daily_high_hours']),
               'exclude_2023_08':primary.loc[primary.index.to_period('M')!=pd.Period('2023-08')]}
    for (scenario,period),group in sensitivity.groupby(['scenario','period']):
        assert set(zip(group.antecedent,group.consequent))==expected_candidates
        for row in group.itertuples():
            if scenario=='pairwise_available_days':
                pair=[row.antecedent,row.consequent]
                z=high.loc[valid[pair].ge(s['minimum_daily_valid_hours']).all(axis=1),pair].ge(s['minimum_daily_high_hours'])
            else:
                z=scenarios[scenario]
            z=z.loc[z.index<split] if period=='discovery' else z.loc[z.index>=split]
            for key,value in manual_metrics(z,row.antecedent,row.consequent).items(): assert_close(getattr(row,key),value)
    seasonal=pd.read_csv(T/'association_rule_seasonal.csv')
    season_map={12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
    for (period,season),group in seasonal.groupby(['period','season']):
        z=train if period=='discovery' else test; z=z.loc[z.index.month.map(season_map)==season]
        for row in group.itertuples():
            for key,value in manual_metrics(z,row.antecedent,row.consequent).items(): assert_close(getattr(row,key),value)
    summary=json.loads((ROOT/'results/checks/association_summary.json').read_text(encoding='utf-8'))
    assert summary['eligible_dates']==len(primary)
    assert summary['discovery_empty_transactions']==int((~train.any(axis=1)).sum())
    assert summary['validation_empty_transactions']==int((~test.any(axis=1)).sum())
    assert summary['candidate_rules']==len(rules) and summary['validation_pass_rules']==int(rules.validation_pass.sum())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==summary['input_sha256']
    checks={'passed':True,'dates_rebuilt':len(days),'station_days_rebuilt':len(counts),'eligible_dates':len(primary),
            'all_directed_pairs_verified':len(pairs),'frequent_itemsets_verified':len(frequent),'candidate_rules_verified':len(rules),
            'discovery_only_selection_checked':True,'empty_transactions_retained':True,'unknowns_not_encoded_as_absence':True,
            'seasonal_and_sensitivity_metrics_checked':True,'bootstrap_interval_bounds_checked':True,
            'known_apriori_fixture_passed':True,'maximum_saved_metric_absolute_error':max(errors),
            'source_hash_unchanged':True}
    (ROOT/'results/checks/association_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
