"""New member-3 analysis. Input files are read only; no old analysis is used."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
import sklearn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
INPUT = (ROOT / CFG['input_directory']).resolve()
TABLES = ROOT / 'results/tables'
FIGS = ROOT / 'results/figures'
CHECKS = ROOT / 'results/checks'
for _d in (TABLES, FIGS, CHECKS):
    _d.mkdir(parents=True, exist_ok=True)
FEATURES = ['overall_risk', 'morning_gap', 'evening_gap', 'weekday_gap',
            'spring_gap', 'autumn_gap', 'winter_gap', 'hourly_range']
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
SEASON_MAP = {12:'Winter', 1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring',
              5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer', 9:'Autumn', 10:'Autumn', 11:'Autumn'}
COUNTS = ['expected', 'present', 'nonmissing', 'excluded_estimated', 'valid', 'high', 'serious']
COLORS = ['#177E89', '#C45B36', '#665A9C', '#8B782E']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.titleweight':'bold', 'savefig.dpi':180})

def save_table(d, name, index=False):
    d.to_csv(TABLES / (name + '.csv'), index=index, encoding='utf-8-sig', float_format='%.12g')

def json_write(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x:
                    x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load_data():
    d = pd.read_csv(INPUT / 'aqhi_long.csv')
    s = pd.read_csv(INPUT / 'stations.csv')
    assert not s.station_code.duplicated().any()
    assert set(d.station) == set(s.station_code)
    assert not d.duplicated(['date', 'hour', 'station']).any()
    assert d.hour.between(1,24).all()
    assert d.aqhi.dropna().isin(range(1,11)).all()
    assert d.loc[d.is_capped.eq(1), 'aqhi'].eq(10).all()
    d['source_datetime'] = pd.to_datetime(d.datetime)
    d['published_datetime'] = pd.to_datetime(d.date) + pd.to_timedelta(d.hour, unit='h')
    assert ((d.published_datetime-d.source_datetime) == pd.Timedelta(hours=1)).all()
    # The source-derived primary timestamp is an explicit interpretation of hour labels.
    d = derive_calendar(d, 'published_datetime')
    d['official_type'] = np.where(d.station.isin(['Central', 'Causeway Bay', 'Mong Kok']), 'Roadside', 'General')
    d['risk_category'] = np.select([d.aqhi.isna(), d.is_capped.eq(1), d.aqhi.ge(8), d.aqhi.eq(7), d.aqhi.ge(4)],
                                  ['Missing','Serious','Very high','High','Moderate'], default='Low')
    d = d.loc[~d.station.isin(CFG['excluded_stations'])].copy()
    s = s.loc[s.station_code.isin(d.station.unique())].copy()
    assert d.station.nunique() == CFG['expected_station_count']
    return d, s

def derive_calendar(d, time_column):
    d = d.copy()
    d['analysis_datetime'] = d[time_column]
    d['month_key'] = d.analysis_datetime.dt.strftime('%Y-%m')
    d['clock_hour'] = d.analysis_datetime.dt.hour
    d['analysis_weekday'] = d.analysis_datetime.dt.dayofweek
    d['analysis_year'] = d.analysis_datetime.dt.year
    d['analysis_season'] = d.analysis_datetime.dt.month.map(SEASON_MAP)
    return d

def make_cube(d, start, end, include_estimated=False):
    """Sufficient counts on an explicit calendar, including entirely absent rows."""
    m = d.loc[d.analysis_datetime.ge(start) & d.analysis_datetime.lt(end)].copy()
    m['present'] = 1
    m['nonmissing'] = m.aqhi.notna().astype('int64')
    m['excluded_estimated'] = (m.aqhi.notna() & m.is_estimated.eq(1) & (not include_estimated)).astype('int64')
    m['valid'] = (m.aqhi.notna() & (include_estimated | m.is_estimated.eq(0))).astype('int64')
    m['high'] = (m.aqhi.ge(CFG['high_risk_threshold']) & m.valid.eq(1)).astype('int64')
    m['serious'] = (m.is_capped.eq(1) & m.valid.eq(1)).astype('int64')
    keys = ['station', 'month_key', 'clock_hour', 'analysis_weekday']
    actual = m.groupby(keys, observed=True)[COUNTS[1:]].sum().reset_index()
    dates = pd.date_range(start, end, freq='h', inclusive='left')
    cal = pd.DataFrame({'month_key':dates.strftime('%Y-%m'), 'clock_hour':dates.hour,
                        'analysis_weekday':dates.dayofweek, 'expected':1})
    cal = cal.groupby(keys[1:]).expected.sum().reset_index()
    grid = pd.DataFrame({'station':sorted(d.station.unique())}).merge(cal, how='cross')
    cube = grid.merge(actual, on=keys, how='left', validate='1:1').fillna(0)
    cube[COUNTS] = cube[COUNTS].astype('int64')
    cube['month_number'] = cube.month_key.str[-2:].astype(int)
    cube['season'] = cube.month_number.map(SEASON_MAP)
    assert (cube.valid <= cube.nonmissing).all() and (cube.high <= cube.valid).all()
    assert (cube.present <= cube.expected).all()
    return cube

def aggregate(c, by):
    g = c.groupby(by, observed=True)[COUNTS].sum().reset_index()
    g['risk'] = g.high / g.valid.replace(0, np.nan)
    g['coverage'] = g.valid / g.expected.replace(0, np.nan)
    return g

def features_from_cube(c, month_weights=None, equal_month=False):
    c = c.copy()
    if month_weights is not None:
        w = c.month_key.map(month_weights).fillna(0)
        c.loc[:,COUNTS] = c[COUNTS].mul(w, axis=0)
    def rate(mask=None):
        z = c if mask is None else c.loc[mask]
        if equal_month:
            g = aggregate(z, ['station', 'month_key'])
            return g.loc[g.coverage.ge(.85)].groupby('station').risk.mean()
        return aggregate(z, ['station']).set_index('station').risk
    f = pd.DataFrame({'overall_risk':rate()})
    ref = rate(c.clock_hour.between(10,16))
    f['morning_gap'] = rate(c.clock_hour.between(7,9)) - ref
    f['evening_gap'] = rate(c.clock_hour.between(17,19)) - ref
    c['daytype'] = np.where(c.analysis_weekday.lt(5), 'Weekday', 'Weekend')
    # Paired station-month contrasts prevent calendar composition imbalance.
    g = aggregate(c, ['station', 'month_key', 'daytype'])
    g.loc[g.coverage.lt(CFG['minimum_month_coverage_for_weekday_gap']), 'risk'] = np.nan
    p = g.pivot(index=['station','month_key'], columns='daytype', values='risk')
    diff = (p.Weekday-p.Weekend).dropna()
    if month_weights is None:
        f['weekday_gap'] = diff.groupby('station').mean()
    else:
        w = diff.index.get_level_values('month_key').map(month_weights).to_numpy(dtype=float)
        f['weekday_gap'] = (diff*w).groupby('station').sum() / pd.Series(w,index=diff.index).groupby('station').sum()
    summer = rate(c.season.eq('Summer'))
    for season in ['Spring','Autumn','Winter']:
        f[season.lower()+'_gap'] = rate(c.season.eq(season))-summer
    if equal_month:
        hr = aggregate(c,['station','month_key','clock_hour'])
        hr = hr.loc[hr.coverage.ge(.85)].groupby(['station','clock_hour']).risk.mean().unstack()
    else:
        hr = aggregate(c,['station','clock_hour']).pivot(index='station',columns='clock_hour',values='risk')
    f['hourly_range'] = hr.max(axis=1)-hr.min(axis=1)
    return f[FEATURES].sort_index()

def fit_clusters(f, k, weights=None, ward=False):
    scaler = StandardScaler()
    x = scaler.fit_transform(f)
    if weights is not None:
        x = x*np.array(weights)
    model = AgglomerativeClustering(n_clusters=k, linkage='ward') if ward else KMeans(
        n_clusters=k, n_init=CFG['n_init'], random_state=CFG['random_seed'])
    raw = model.fit_predict(x)
    # Stable display labels, ordered by increasing overall risk; analysis uses ARI.
    order = sorted(np.unique(raw), key=lambda a: f.loc[raw==a,'overall_risk'].mean())
    labels = pd.Series([order.index(a)+1 for a in raw],index=f.index,name='cluster')
    return labels, x, scaler

def feature_coverage(c):
    masks = {'overall':np.ones(len(c),bool), 'morning':c.clock_hour.between(7,9),
             'daytime':c.clock_hour.between(10,16), 'evening':c.clock_hour.between(17,19),
             'weekday':c.analysis_weekday.lt(5), 'weekend':c.analysis_weekday.ge(5)}
    masks.update({s:c.season.eq(s) for s in SEASONS})
    out=[]
    for name,mask in masks.items():
        g=aggregate(c.loc[mask],['station']); g['feature_group']=name; out.append(g)
    hours=aggregate(c,['station','clock_hour']); hours['feature_group']='hour_'+hours.clock_hour.astype(str)
    return pd.concat(out+[hours],ignore_index=True)

def plot_save(fig,name):
    fig.savefig(FIGS/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIGS/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    print('Loading member-2 files...',flush=True)
    d,s=load_data()
    manifest=pd.DataFrame([{'file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)}
                          for p in sorted(INPUT.iterdir()) if p.is_file()])
    save_table(manifest,'input_manifest')
    save_table(s,'station_metadata')
    audit={'rows':len(d),'stations':d.station.nunique(),'source_files':d.source_file.nunique(),
           'source_datetime_min':str(d.source_datetime.min()),'source_datetime_max':str(d.source_datetime.max()),
           'analysis_datetime_min':str(d.analysis_datetime.min()),'analysis_datetime_max':str(d.analysis_datetime.max()),
           'missing':int(d.aqhi.isna().sum()),'estimated_nonmissing':int((d.aqhi.notna() & d.is_estimated.eq(1)).sum()),
           'capped':int(d.is_capped.sum()),'duplicate_source_keys':int(d.duplicated(['date','hour','station']).sum())}
    save_table(d.loc[d.aqhi.notna()].groupby('station').analysis_datetime.agg(['min','max','size']).reset_index(),'station_availability')
    json_write(audit,CHECKS/'input_audit.json')
    c=make_cube(d,CFG['main_start'],CFG['main_end_exclusive'])
    save_table(c,'monthly_cube')
    cover=aggregate(c,['station']); cover['absent']=cover.expected-cover.present
    save_table(cover,'station_coverage')
    save_table(aggregate(c,['station','month_key']),'monthly_coverage')
    fc=feature_coverage(c); save_table(fc,'feature_coverage')
    eligible = cover.loc[cover.coverage.ge(CFG['minimum_overall_coverage']),'station']
    eligible = sorted(set(eligible) & set(fc.groupby('station').coverage.min().loc[lambda x:x.ge(CFG['minimum_feature_coverage'])].index))
    f=features_from_cube(c).loc[eligible]
    assert f.notna().all().all()
    save_table(f,'station_features',True)
    selection=[]
    for k in CFG['k_candidates']:
        lab,x,_=fit_clusters(f,k)
        selection.append({'k':k,'silhouette':silhouette_score(x,lab),'min_cluster_size':lab.value_counts().min(),
                          'sizes':';'.join(map(str,lab.value_counts().sort_index().tolist()))})
    sel=pd.DataFrame(selection); save_table(sel,'model_selection')
    # Predeclared internal selection, with stability and size reported independently.
    k=int(sel.loc[sel.silhouette.idxmax(),'k'])
    labels,x,scaler=fit_clusters(f,k)
    save_table(pd.DataFrame(x,index=f.index,columns=FEATURES),'standardized_features',True)
    json_write({'features':FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},CHECKS/'scaler.json')
    assignments=labels.to_frame().join(cover.set_index('station')[['risk','coverage','high','valid','expected']])
    save_table(assignments,'station_clusters',True)
    cent=f.join(labels).groupby('cluster').mean(); save_table(cent,'cluster_centres',True)
    profiles={}
    for name,group in [('hourly','clock_hour'),('seasonal','season'),('weekly','analysis_weekday')]:
        p=aggregate(c,['station',group]); save_table(p,'station_'+name)
        p=p.merge(labels,on='station',how='inner')
        g=p.groupby(['cluster',group]).risk.mean().reset_index()
        profiles[name]=g; save_table(g,'cluster_'+name)
    # Export representative period risks, unadjusted weekday risks and category shares.
    cp=c.copy(); cp['period']=pd.cut(cp.clock_hour,[-1,6,9,16,19,23],labels=['00-06','07-09','10-16','17-19','20-23'])
    save_table(aggregate(cp,['station','period']),'station_periods')
    m=d.loc[d.analysis_datetime.ge(CFG['main_start']) & d.analysis_datetime.lt(CFG['main_end_exclusive']) & d.aqhi.notna() & d.is_estimated.eq(0)]
    save_table(m.groupby(['station','risk_category']).size().rename('count').reset_index(),'risk_categories')
    # Sensitivity analyses: common station set and fixed primary K.
    checks=[]; alt_assign=labels.rename('primary').to_frame()
    def compare(name,ff,weights=None,ward=False):
        nonlocal alt_assign
        ff=ff.reindex(f.index).dropna()
        la,xx,_=fit_clusters(ff,k,weights,ward)
        checks.append({'scenario':name,'stations':len(ff),'ari':adjusted_rand_score(labels.loc[ff.index],la),
                       'silhouette':silhouette_score(xx,la)})
        alt_assign=alt_assign.join(la.rename(name))
    compare('include_estimated',features_from_cube(make_cube(d,CFG['main_start'],CFG['main_end_exclusive'],True)))
    compare('exclude_2023_08',features_from_cube(c.loc[c.month_key.ne('2023-08')]))
    alternate=derive_calendar(d,'source_datetime')
    compare('delivered_time',features_from_cube(make_cube(alternate,CFG['main_start'],CFG['main_end_exclusive'])))
    period_models=[]
    for name,start,end in CFG['period_checks']:
        ff=features_from_cube(c.loc[c.month_key.ge(start) & c.month_key.lt(end)])
        save_table(ff, 'period_features_'+name, True)
        compare(name,ff)
        for kk in CFG['k_candidates']:
            la,xx,_=fit_clusters(ff,kk)
            period_models.append({'period':name,'k':kk,'silhouette':silhouette_score(xx,la),
                                  'sizes':';'.join(map(str,la.value_counts().sort_index().tolist()))})
            save_table(la.to_frame(),'period_labels_'+name+'_k'+str(kk),True)
    save_table(pd.DataFrame(period_models),'period_model_selection')
    compare('ward',f,ward=True)
    compare('equal_feature_blocks',f,weights=[1,1/np.sqrt(3),1/np.sqrt(3),1,1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
    compare('equal_months',features_from_cube(c,equal_month=True))
    stricter=fc.groupby('station').coverage.min(); ff=f.loc[stricter.reindex(f.index).ge(.90)]
    if len(ff)>k: compare('minimum_group_coverage_90pct',ff)
    save_table(pd.DataFrame(checks),'robustness'); save_table(alt_assign,'robustness_assignments',True)
    print('Primary clustering:',k,'groups; bootstrap starts...',flush=True)
    rng=np.random.default_rng(CFG['random_seed']); months=sorted(c.month_key.unique())
    B=CFG['bootstrap_repetitions']; co=np.zeros((len(f),len(f))); bs=[]; bf=[]
    for b in range(B):
        draws=[]
        for month in range(1,13):
            pool=[a for a in months if int(a[-2:])==month]
            draws.extend(rng.choice(pool,len(pool),replace=True).tolist())
        weights=pd.Series(draws).value_counts().to_dict()
        fb=features_from_cube(c,weights).reindex(f.index)
        if fb.isna().any().any(): raise ValueError('Undefined bootstrap feature')
        lb,_,_=fit_clusters(fb,k)
        ari=adjusted_rand_score(labels,lb); bs.append({'replicate':b,'ari':ari})
        co += np.equal.outer(lb.to_numpy(),lb.to_numpy())
        bf.append(fb.to_numpy())
        if (b+1)%50==0: print('Bootstrap',b+1,'/',B,flush=True)
    save_table(pd.DataFrame(bs),'bootstrap_ari')
    save_table(pd.DataFrame(co/B,index=f.index,columns=f.index),'bootstrap_coassignment',True)
    arr=np.array(bf); ci=[]
    for i,station in enumerate(f.index):
        for j,feature in enumerate(FEATURES):
            lo,hi=np.quantile(arr[:,i,j],[.025,.975])
            ci.append({'station':station,'feature':feature,'estimate':f.iloc[i,j],'lower':lo,'upper':hi})
    save_table(pd.DataFrame(ci),'feature_bootstrap_intervals')
    # Fixed early panel: all input stations except the three late additions.
    early=sorted(set(d.station)-{'Tseung Kwan O','Southern','North'})
    long=make_cube(d,'2014-01-01','2025-01-01')
    long=long.loc[long.station.isin(early)]; long['year']=long.month_key.str[:4].astype(int)
    annual=aggregate(long,['station','year']); save_table(annual,'longterm_station_annual')
    panel=annual.groupby('year').agg(risk=('risk','mean'),coverage=('coverage','mean'),stations=('station','nunique')).reset_index()
    save_table(panel,'longterm_panel_annual')
    # Recent period is intentionally separated; incomplete edge months are excluded.
    recent=make_cube(d,'2025-04-01','2026-07-01'); save_table(aggregate(recent,['station','month_key']),'recent_post_update_monthly')
    # Scientific figures, separate from native editable slide charts.
    order=assignments.sort_values(['cluster','risk']).index
    fig,ax=plt.subplots(figsize=(11,7)); im=ax.imshow(pd.DataFrame(x,index=f.index).loc[order],aspect='auto',cmap='RdBu_r',vmin=-2.5,vmax=2.5)
    ax.set_yticks(range(len(order)),[f'{a}  [C{labels[a]}]' for a in order]); ax.set_xticks(range(len(FEATURES)),FEATURES,rotation=35,ha='right')
    ax.set_title('Station risk profiles, 2014-2024 (15 stations)'); fig.colorbar(im,ax=ax,label='Standard deviations'); fig.tight_layout(); plot_save(fig,'feature_heatmap')
    fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    axs[0].plot(sel.k,sel.silhouette,'o-',color=COLORS[0]); axs[0].set(xlabel='Number of clusters',ylabel='Silhouette',xticks=sel.k,title='K-means model selection')
    dendrogram(linkage(x,method='ward'),labels=f.index.tolist(),ax=axs[1],leaf_rotation=75,leaf_font_size=8)
    axs[1].set_title('Ward hierarchy'); fig.tight_layout(); plot_save(fig,'model_selection')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for cl in sorted(labels.unique()):
        a=profiles['hourly'].query('cluster==@cl'); axs[0].plot(a.clock_hour,a.risk*100,label=f'Cluster {cl}',color=COLORS[cl-1],linewidth=2)
        a=profiles['seasonal'].query('cluster==@cl').set_index('season').reindex(SEASONS)
        axs[1].plot(SEASONS,a.risk*100,'o-',label=f'Cluster {cl}',color=COLORS[cl-1],linewidth=2)
    axs[0].set(xlabel='Publication clock hour (HKT)',ylabel='High-risk hours (%)',xticks=range(0,24,3),title='Daily profiles')
    axs[1].set(ylabel='High-risk hours (%)',title='Seasonal profiles')
    for ax in axs: ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); plot_save(fig,'cluster_profiles')
    fig,ax=plt.subplots(figsize=(9,6)); a=assignments.sort_values('risk')
    ax.barh(a.index,a.risk*100,color=[COLORS[v-1] for v in a.cluster]); ax.set(xlabel='High-risk hours (%)',title='Station risk, 2014-2024 (15 stations)')
    fig.tight_layout(); plot_save(fig,'station_risk')
    fig,ax=plt.subplots(figsize=(8.5,4)); ax.plot(panel.year,panel.risk*100,'o-',color=COLORS[0]); ax.set(xlabel='Year',ylabel='Equal-station high-risk hours (%)',title='Fixed 15-station panel, 2014-2024',xticks=panel.year)
    ax.grid(axis='y',alpha=.2); fig.tight_layout(); plot_save(fig,'longterm_panel')
    fig,ax=plt.subplots(figsize=(8,6)); mat=pd.DataFrame(co/B,index=f.index,columns=f.index).loc[order,order]
    im=ax.imshow(mat,vmin=0,vmax=1,cmap='YlGnBu'); ax.set_xticks(range(len(order)),order,rotation=90,fontsize=8); ax.set_yticks(range(len(order)),order,fontsize=8)
    ax.set_title(f'Co-clustering over {B} month-block resamples'); fig.colorbar(im,ax=ax,label='Fraction of resamples'); fig.tight_layout(); plot_save(fig,'cluster_stability')
    mon=aggregate(c,['station','month_key']); grid=mon.pivot(index='station',columns='month_key',values='coverage')
    fig,ax=plt.subplots(figsize=(11,5)); im=ax.imshow(grid,vmin=.7,vmax=1,cmap='YlGnBu',aspect='auto'); ax.set_yticks(range(len(grid)),grid.index,fontsize=8)
    ax.set_xticks(range(0,len(grid.columns),6),grid.columns[::6],rotation=35); ax.set_title('Analysis coverage by station and month'); fig.colorbar(im,ax=ax,label='Valid / expected'); fig.tight_layout(); plot_save(fig,'coverage')
    stats={'period':'2014-2024','station_count':len(f),'excluded_stations':CFG['excluded_stations'],'selected_k':k,'silhouette':float(sel.loc[sel.k.eq(k),'silhouette'].iloc[0]),'eligible_stations':eligible,
           'expected':int(c.expected.sum()),'present':int(c.present.sum()),'nonmissing':int(c.nonmissing.sum()),
           'excluded_estimated':int(c.excluded_estimated.sum()),'valid':int(c.valid.sum()),'high':int(c.high.sum()),
           'analysis_coverage':float(c.valid.sum()/c.expected.sum()),'bootstrap_ari_median':float(np.median([a['ari'] for a in bs])),
           'bootstrap_exact_fraction':float(np.mean([a['ari']==1 for a in bs])),
           'bootstrap_ari_q025':float(np.quantile([a['ari'] for a in bs],.025)),
           'bootstrap_ari_q975':float(np.quantile([a['ari'] for a in bs],.975)),
           'clusters':{str(i):labels.loc[labels.eq(i)].index.tolist() for i in sorted(labels.unique())},
           'cluster_risk':{str(i):float(v) for i,v in f.join(labels).groupby('cluster').overall_risk.mean().items()},
           'runtime':{'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__,
                      'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    json_write(stats,CHECKS/'analysis_summary.json')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
