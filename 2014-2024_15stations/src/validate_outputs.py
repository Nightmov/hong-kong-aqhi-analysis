"""Recalculate results from source CSV, independently of feature-building code."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    inp=(ROOT/cfg['input_directory']).resolve(); tab=ROOT/'results/tables'
    d=pd.read_csv(inp/'aqhi_long.csv'); t=pd.to_datetime(d.date)+pd.to_timedelta(d.hour,unit='h')
    d['t']=t; d['hour0']=t.dt.hour; d['month0']=t.dt.month
    d['weekend0']=t.dt.dayofweek.ge(5); d['monthkey0']=t.dt.strftime('%Y-%m')
    m=d.loc[t.ge(cfg['main_start']) & t.lt(cfg['main_end_exclusive']) & ~d.station.isin(cfg['excluded_stations'])].copy()
    assert m.station.nunique()==cfg['expected_station_count']
    ok=m.aqhi.notna() & m.is_estimated.eq(0); v=m.loc[ok].copy(); v['high0']=v.aqhi.ge(7)
    f=pd.read_csv(tab/'station_features.csv',index_col='station')
    rebuilt=[]
    for station in f.index:
        q=v.loc[v.station.eq(station)]
        risk=lambda a: a.high0.mean()
        overall=risk(q); daytime=risk(q.loc[q.hour0.between(10,16)])
        morning=risk(q.loc[q.hour0.between(7,9)])-daytime
        evening=risk(q.loc[q.hour0.between(17,19)])-daytime
        differences=[]
        for month,qq in q.groupby('monthkey0'):
            calendar=pd.date_range(month+'-01',pd.Timestamp(month+'-01')+pd.offsets.MonthBegin(),freq='h',inclusive='left')
            expected=[int((calendar.dayofweek<5).sum()),int((calendar.dayofweek>=5).sum())]
            sub=[qq.loc[~qq.weekend0],qq.loc[qq.weekend0]]
            if all(len(sub[j])/expected[j]>=cfg['minimum_month_coverage_for_weekday_gap'] for j in [0,1]):
                differences.append(risk(sub[0])-risk(sub[1]))
        summer=risk(q.loc[q.month0.isin([6,7,8])])
        seasonal=[risk(q.loc[q.month0.isin(months)])-summer for months in [[3,4,5],[9,10,11],[12,1,2]]]
        hr=q.groupby('hour0').high0.mean()
        rebuilt.append([overall,morning,evening,np.mean(differences),*seasonal,hr.max()-hr.min()])
    independent=pd.DataFrame(rebuilt,index=f.index,columns=f.columns)
    error=float(np.max(np.abs(f.to_numpy()-independent.to_numpy())))
    assert error<1e-10, error
    manifest=pd.read_csv(tab/'input_manifest.csv')
    for row in manifest.itertuples():
        assert hashlib.sha256((inp/row.file).read_bytes()).hexdigest()==row.sha256
    cube=pd.read_csv(tab/'monthly_cube.csv'); coverage=pd.read_csv(tab/'station_coverage.csv').set_index('station')
    expected=len(pd.date_range(cfg['main_start'],cfg['main_end_exclusive'],freq='h',inclusive='left'))
    assert int(cube.expected.sum())==expected*cfg['expected_station_count']
    assert int(cube.present.sum())==len(m)
    assert int(cube.nonmissing.sum())==int(m.aqhi.notna().sum())
    assert int(cube.valid.sum())==len(v) and int(cube.high.sum())==int(v.high0.sum())
    assert int((coverage.expected-coverage.present).sum())==int(cube.expected.sum())-len(m)
    assert not d.duplicated(['station','t']).any()
    # Real boundary examples, all months: source 24 becomes next calendar day at 00.
    end=d.loc[d.hour.eq(24)]
    assert end.t.dt.hour.eq(0).all()
    assert (end.t.dt.normalize()==pd.to_datetime(end.date)+pd.Timedelta(days=1)).all()
    categories=pd.read_csv(tab/'risk_categories.csv')
    assert categories['count'].sum()==len(v)
    serious=int(categories.loc[categories.risk_category.eq('Serious'),'count'].sum())
    assert serious==int(v.is_capped.sum())
    co=pd.read_csv(tab/'bootstrap_coassignment.csv',index_col=0).to_numpy()
    assert np.allclose(co,co.T) and np.allclose(np.diag(co),1) and ((co>=0)&(co<=1)).all()
    independent.to_csv(tab/'independently_recomputed_features.csv',encoding='utf-8-sig')
    checks={'passed':True,'independently_recalculated_features':int(f.size),'feature_max_abs_error':error,
            'input_hashes_unchanged':True,'calendar_expected_checked':True,'count_conservation_checked':True,
            'midnight_rollover_checked':True,'ten_plus_category_checked':True,'bootstrap_matrix_checked':True,
            'analysis_rows':len(v),'expected_station_hours':expected*cfg['expected_station_count']}
    (ROOT/'results/checks/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
