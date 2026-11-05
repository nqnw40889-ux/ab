"""Nhóm 16 · AI4I research demo. Inference only; no training on the web."""
from pathlib import Path
from datetime import datetime, timezone
import io, csv, hashlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from core import FEATURES, NUM, LABELS, predict_batch
from evaluation import measure

ROOT = Path(__file__).parent
st.set_page_config(page_title='Nhóm 16 | Machine Health', page_icon='⚙️', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'Be Vietnam Pro',sans-serif}
.stApp{background:#f3f6fa;color:#183449}.block-container{max-width:1250px;padding-top:2rem}
h1,h2,h3{letter-spacing:-.035em;color:#12374a} [data-testid="stSidebar"]{background:#102c3b}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h3{color:#e4f0f3!important}
.hero{background:linear-gradient(110deg,#102f40,#155568);border-radius:22px;padding:32px 36px;color:white;margin-bottom:24px;position:relative;overflow:hidden}
.hero:after{content:'⚙';position:absolute;right:28px;top:-24px;font-size:200px;color:#ffffff0b;line-height:1.2}
.hero h1{color:white;font-size:32px;margin:10px 0}.hero p{color:#c7dce5;max-width:750px;margin-bottom:0}.eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:#76e0cb}
.card{background:white;border:1px solid #e0e8ef;border-radius:16px;padding:24px;min-height:165px;margin:12px 0}.card h3{font-size:19px}.card p{color:#566c7b;font-size:14px}
[data-testid="stMetric"]{background:white;border:1px solid #dfe8ef;border-radius:15px;padding:18px}
[data-testid="stMetricValue"]{font-size:27px} [data-testid="stForm"]{background:white;border:1px solid #dce6ee;border-radius:18px;padding:24px}
.stButton>button,.stDownloadButton>button{border-radius:10px}div[data-testid="stExpander"]{background:white;border-radius:12px}
.footer{font-size:12px;color:#647c8b;border-top:1px solid #dce5ed;padding-top:15px;margin-top:36px}
@media(max-width:700px){.hero{padding:22px}.hero h1{font-size:25px}.block-container{padding:1rem}}
</style>''', unsafe_allow_html=True)
PAGES=['Tổng quan','Dự đoán một mẫu','CSV hàng loạt','Kiểm chứng test','So sánh mô hình','Lịch sử','Thông tin đề tài']
st.sidebar.title('⚙️ MACHINE HEALTH')
st.sidebar.caption('NHÓM 16  /  HỌC MÁY')
page=st.sidebar.radio('Điều hướng',PAGES)
st.sidebar.divider()
st.sidebar.markdown('### AI4I 2020')
st.sidebar.caption('Dự đoán hỏng hóc máy phục vụ bảo trì dự đoán trên dữ liệu cảm biến mất cân bằng.')
st.sidebar.caption('Demo nghiên cứu • Dữ liệu mô phỏng')

@st.cache_resource
def load_model(stamp):
    return joblib.load(ROOT/'mo_hinh_demo.joblib')
try:
    bundle=load_model((ROOT/'mo_hinh_demo.joblib').stat().st_mtime_ns)
    meta=bundle['metadata']
except Exception as e:
    st.error('Không tải được mô hình. Đặt mo_hinh_demo.joblib cạnh app.py và cài đúng requirements.txt.');st.code(str(e));st.stop()
threshold=float(bundle['threshold'])
st.session_state.setdefault('history',[])

def download(df,name,label='Tải kết quả CSV',key=None):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv',key=key)

def remember(result,source):
    rows=result[result.prediction.notna()].copy()
    rows['timestamp_utc']=datetime.now(timezone.utc).isoformat();rows['source']=source
    st.session_state.history=(st.session_state.history+rows.to_dict('records'))[-1000:]

def explain(x):
    pipe=bundle.get('explanation_pipeline',bundle['pipeline'])
    if not hasattr(pipe,'steps') or pipe.steps[-1][1].__class__.__name__!='LGBMClassifier':
        st.info('Giải thích đóng góp hiện hỗ trợ mô hình nền LightGBM.');return
    prep,model=pipe.steps[0][1],pipe.steps[-1][1]
    z=prep.transform(x[FEATURES]);values=np.asarray(model.booster_.predict(z,pred_contrib=True))[0]
    raw_score = float(model.booster_.predict(z, raw_score=True)[0])
    if not np.isclose(raw_score, values.sum(), atol=1e-8):
        st.warning('Chưa thể hiển thị giải thích cho mẫu này.')
        return
    names=[n.split('__')[-1] for n in prep.get_feature_names_out()]
    names=[LABELS.get(n,{'Type_M':'Loại sản phẩm M','Type_H':'Loại sản phẩm H'}.get(n,n)) for n in names]
    table=pd.DataFrame({'Thuộc tính':names,'Đóng góp log-odds':values[:-1]}).sort_values('Đóng góp log-odds')
    fig,ax=plt.subplots(figsize=(9,4));v=table['Đóng góp log-odds']
    ax.barh(table['Thuộc tính'],v,color=['#e18345' if a>0 else '#168b91' for a in v]);ax.axvline(0,color='#9baab4',lw=.8)
    ax.set_xticks([])
    ax.spines[['top','right','bottom','left']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Thanh cam hướng sang phải làm tăng điểm dự đoán hỏng. Thanh xanh hướng sang trái làm giảm điểm. Thanh dài hơn thể hiện ảnh hưởng lớn hơn trong mẫu này.')

def predict_input(frame):
    result = predict_batch(bundle, frame, meta, allow_outside=False)
    result['Kết luận'] = result.prediction.map({0: 'Dự đoán không hỏng', 1: 'Dự đoán có hỏng'}).astype('string')
    result.loc[result.range_warning.ne(''), 'Kết luận'] = 'Ngoài phạm vi train — không đủ cơ sở kết luận'
    result.loc[result.input_error.ne(''), 'Kết luận'] = 'Dữ liệu không hợp lệ'
    return result

def show_result(result):
    row=result.iloc[0]
    if row.input_error:st.error(row.input_error);return
    if row.range_warning:
        st.warning('Ngoài phạm vi train — không đủ cơ sở kết luận.')
        ranges = meta['train_ranges']
        for col in NUM:
            value = float(row[col])
            lower, upper = ranges[col]['min'], ranges[col]['max']
            if not lower <= value <= upper:
                st.write(f'{LABELS[col]} · Đã nhập **{value:g}** · Khoảng huấn luyện **{lower:g} – {upper:g}**')
        return
    if row.prediction==1:st.error('MÔ HÌNH DỰ ĐOÁN: CÓ HỎNG • Nhãn 1')
    else:st.success('MÔ HÌNH DỰ ĐOÁN: KHÔNG HỎNG • Nhãn 0')
    a,b=st.columns(2);a.metric('Điểm xác suất lớp hỏng',f'{row.probability:.2%}');b.metric('Ngưỡng quyết định',f'{threshold:.2%}')
    st.caption('Điểm từ ngưỡng trở lên được xếp vào lớp hỏng. Điểm này do mô hình ước lượng trên dữ liệu mô phỏng.')
    with st.expander('Thông số của lần dự đoán này'):st.dataframe(result[FEATURES],hide_index=True,width='stretch')
    with st.expander('Giải thích đóng góp của từng thuộc tính'):explain(result[FEATURES])
    download(result,'du_doan_mot_mau.csv')

st.markdown(f'<div class="hero"><div class="eyebrow">NHÓM 16 · PREDICTIVE MAINTENANCE</div><h1>{page}</h1><p>Phân loại trạng thái máy từ thông số vận hành. Kiểm chứng kết quả bằng dữ liệu có nhãn.</p></div>',unsafe_allow_html=True)

if page=='Tổng quan':
    a,b,c,d=st.columns(4);a.metric('Mô hình',bundle['model_name']);b.metric('Ngưỡng',f'{threshold:.2%}');c.metric('Dữ liệu AI4I','10.000 mẫu');d.metric('Đầu vào','6 thuộc tính')
    metrics=pd.read_csv(ROOT/'reports/metrics.csv');r=metrics.set_index('Split').loc['Test']
    st.subheader('Kết quả kiểm thử đã ghi nhận')
    detected, missed, false_alarm = st.columns(3)
    detected.metric('Phát hiện đúng mẫu hỏng', f'{int(r.TP)} / {int(r.TP+r.FN)}')
    missed.metric('Mẫu hỏng bị bỏ sót', int(r.FN))
    false_alarm.metric('Cảnh báo nhầm', int(r.FP))
    st.caption('Đây là kết quả trên tập test của AI4I. Tập test đã được xem trong các vòng phát triển; không phải đánh giá độc lập trên máy thật.')

elif page=='Dự đoán một mẫu':
    sample=st.selectbox('Chọn dữ liệu minh họa',['Tự nhập','Mẫu không hỏng','Mẫu có hỏng'])
    defaults={'Tự nhập':('M',300.,310.,1500,40.,100),'Mẫu không hỏng':('M',302.5,312.,1487,46.1,175),'Mẫu có hỏng':('L',304.,312.9,1363,62.5,200)}[sample]
    if st.session_state.get('last_sample')!=sample:
        st.session_state.pop('single_result',None);st.session_state.last_sample=sample
    st.caption('Hai mẫu có sẵn thuộc validation và được chọn trong các mẫu dự đoán đúng. Chúng chỉ minh họa thao tác.')
    with st.form('sample_'+sample):
        a,b=st.columns(2)
        typ=a.selectbox('Loại sản phẩm (Type)',['L','M','H'],index=['L','M','H'].index(defaults[0]))
        air=a.number_input(LABELS[NUM[0]],value=defaults[1],step=.1)
        proc=a.number_input(LABELS[NUM[1]],value=defaults[2],step=.1)
        speed=b.number_input(LABELS[NUM[2]],value=defaults[3],step=1)
        torque=b.number_input(LABELS[NUM[3]],value=defaults[4],step=.1)
        wear=b.number_input('Độ mòn dụng cụ (phút sử dụng)',value=defaults[5],step=1)
        submit=st.form_submit_button('Phân tích trạng thái',type='primary')
    if submit:
        x=pd.DataFrame([[typ,air,proc,speed,torque,wear]],columns=FEATURES)
        r=predict_input(x);st.session_state.single_result=r;remember(r,'single')
    if 'single_result' in st.session_state:
        st.caption('Kết quả lần bấm gần nhất. Sau khi sửa thông số, bấm Phân tích trạng thái để cập nhật.')
        show_result(st.session_state.single_result)
    with st.expander('Phạm vi quan sát trên tập train'):
        st.dataframe(pd.DataFrame(meta['train_ranges']).T.rename(columns={'min':'Nhỏ nhất','max':'Lớn nhất'}),width='stretch')
        st.caption('Khoảng quan sát của từng biến, không phải giới hạn an toàn. Ngoài khoảng sẽ tự cảnh báo; trong khoảng không bảo đảm tổ hợp hợp lý hoặc dự đoán đúng.')

elif page=='CSV hàng loạt':
    st.write('Mỗi dòng là một mẫu, gồm 6 cột đầu vào. Nhãn và mã định danh nếu có chỉ được giữ để đối chiếu, không đưa vào mô hình.')
    template=pd.read_csv(ROOT/'data/mau_nhap.csv');download(template,'mau_nhap.csv','Tải file CSV mẫu')
    upload=st.file_uploader('Tải CSV (tối đa 10 MB / 10.000 dòng)',type='csv')
    fingerprint=hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
    if st.session_state.get('upload_hash')!=fingerprint:
        st.session_state.pop('batch_result',None);st.session_state.upload_hash=fingerprint
    if st.button('Dự đoán từ CSV',type='primary'):
        if upload is None:st.warning('Hãy chọn file CSV trước.')
        else:
            st.session_state.pop('batch_result',None)
            try:
                if upload.size>10*1024*1024:raise ValueError('Tệp vượt 10 MB.')
                text=upload.getvalue().decode('utf-8-sig');header=[v.strip() for v in next(csv.reader(io.StringIO(text)))]
                if len(header)!=len(set(header)):raise ValueError('Tên cột trùng nhau.')
                frame=pd.read_csv(io.StringIO(text));frame.columns=frame.columns.str.strip()
                if not 1<=len(frame)<=10000:raise ValueError('Cần từ 1 đến 10.000 dòng.')
                r=predict_input(frame);st.session_state.batch_result=r;remember(r,'csv')
            except Exception as e:st.error('Không xử lý được CSV: '+str(e))
    if 'batch_result' in st.session_state:
        r=st.session_state.batch_result;a,b,c=st.columns(3)
        a.metric('Đã dự đoán',int(r.prediction.notna().sum()));b.metric('Dòng lỗi',int(r.input_error.ne('').sum()));c.metric('Ngoài phạm vi train',int(r.range_warning.ne('').sum()))
        st.dataframe(r,hide_index=True,width='stretch');download(r,'ket_qua_csv.csv')
        st.caption('Dòng lỗi vẫn được giữ, nhãn dự đoán để trống. Với dòng ngoài phạm vi train, cột Kết luận ghi lý do và để trống nhãn cùng xác suất.')
    with st.expander('Nếu Excel hiển thị cả dòng trong một cột'):
        st.write('CSV mẫu dùng dấu phẩy phân cách và dấu chấm thập phân. Trong Excel chọn Data → From Text/CSV và chọn dấu phẩy. Không đổi tên cột hoặc lưu thành XLSX để tải vào chức năng này.')
        st.write('Tải ai4i2020.csv sẽ xử lý cả 10.000 mẫu, trong đó có dữ liệu train. Không dùng kết quả đó làm độ chính xác kiểm thử độc lập.')

elif page=='Kiểm chứng test':
    st.write('Dự đoán lại 1.500 mẫu test bằng mô hình đang dùng trên web rồi đối chiếu với nhãn Machine failure. Không điều chỉnh mô hình hoặc ngưỡng tại đây.')
    test=pd.read_csv(ROOT/'data/test_goc.csv');download(test,'test_goc.csv','Tải tập test có nhãn')
    st.caption('Tập test này đã được quan sát ở các vòng phát triển trước. Việc kiểm chứng ở đây xác nhận tính nhất quán của chương trình.')
    if st.button('Chạy kiểm chứng 1.500 mẫu',type='primary'):
        r=predict_batch(bundle,test,meta)
        r['actual']=test['Machine failure'];r['correct']=r.prediction.eq(r.actual)
        r['Trường hợp']=np.select([(r.actual==0)&(r.prediction==0),(r.actual==1)&(r.prediction==1),(r.actual==1)&(r.prediction==0)],['Đúng · Không hỏng','Đúng · Có hỏng','Sai · Bỏ sót'],default='Sai · Cảnh báo nhầm')
        st.session_state.test_result=r
    if 'test_result' in st.session_state:
        r=st.session_state.test_result
        if r.prediction.isna().any():st.error('Có mẫu không dự đoán được. Chưa thể đánh giá toàn bộ tập test.');st.stop()
        m=measure(r.actual,r.probability,threshold)
        a,b,c,d=st.columns(4);a.metric('Precision',f'{m["Precision"]:.2%}');b.metric('Recall',f'{m["Recall"]:.2%}');c.metric('F1',f'{m["F1"]:.4f}');d.metric('AP',f'{m["AP"]:.4f}')
        st.write(f'Phát hiện {m["TP"]}/{m["TP"]+m["FN"]} mẫu hỏng; bỏ sót {m["FN"]}; cảnh báo nhầm {m["FP"]}. Accuracy: {m["Accuracy"]:.2%}.')
        cm=pd.DataFrame([[m['TN'],m['FP']],[m['FN'],m['TP']]],index=['Thực tế không hỏng','Thực tế có hỏng'],columns=['Dự đoán không hỏng','Dự đoán có hỏng'])
        st.dataframe(cm,width='stretch')
        expected=pd.read_csv(ROOT/'reports/predictions_test.csv').set_index('source_row')
        ref=expected.loc[r.source_row,'probability'].to_numpy();error=float(np.max(np.abs(ref-r.probability.to_numpy())))
        if error<1e-10:st.success(f'Khớp kết quả thực nghiệm đã lưu. Sai lệch xác suất lớn nhất: {error:.2e}.')
        else:st.error(f'Không khớp kết quả thực nghiệm đã lưu: sai lệch {error:.2e}. Kiểm tra phiên bản mô hình.')
        choice=st.selectbox('Lọc trường hợp',['Tất cả','Đúng · Không hỏng','Đúng · Có hỏng','Sai · Bỏ sót','Sai · Cảnh báo nhầm'])
        view=r if choice=='Tất cả' else r[r['Trường hợp']==choice]
        st.dataframe(view,hide_index=True,width='stretch');download(r,'kiem_chung_test.csv')
        st.caption('Nhãn thật là nhãn có sẵn trong dataset mô phỏng, không phải kết luận từ kiểm tra máy thật.')

elif page=='So sánh mô hình':
    st.write('Chọn mô hình bằng Average Precision trên validation; chọn ngưỡng tối đa F1 trên validation. Tiền xử lý và huấn luyện chỉ dùng train.')
    tabs=st.tabs(['Mô hình & ngưỡng','Kết quả & biểu đồ'])
    with tabs[0]:
        df = pd.read_csv(ROOT/'reports/validation_comparison.csv')
        st.dataframe(df[['Model', 'AP', 'Precision', 'Recall', 'F1']], hide_index=True, width='stretch')
        download(df, 'so_sanh_validation.csv', 'Tải bảng đầy đủ')
        st.info(f'Mô hình đang triển khai: {bundle["model_name"]}. Ngưỡng: {threshold:.6f}. Các dòng so sánh ban đầu dùng ngưỡng 0,5.')
        st.write('Accuracy: tỷ lệ đúng tổng thể. Precision: trong các cảnh báo hỏng, bao nhiêu là đúng. Recall: trong các mẫu hỏng, phát hiện được bao nhiêu. F1 cân bằng Precision và Recall. AP đánh giá khả năng xếp hạng qua nhiều ngưỡng.')
    with tabs[1]:
        st.dataframe(pd.read_csv(ROOT/'reports/metrics.csv'),hide_index=True,width='stretch')
        a,b=st.columns(2);a.image(str(ROOT/'reports/pr_curve.png'));b.image(str(ROOT/'reports/overfit.png'))
        st.image(str(ROOT/'reports/confusion_test.png'),width=500)
        st.caption('AP khác PR-AUC hình thang. Kết quả train cao hơn validation cần được phân tích; không chỉ dựa vào Accuracy để kết luận tốt với dữ liệu mất cân bằng.')
elif page=='Lịch sử':
    st.caption('Lưu tối đa 1.000 dự đoán gần nhất của phiên. Đóng phiên hoặc khởi động lại có thể mất lịch sử; tải CSV để lưu lâu dài. Kiểm chứng test không ghi vào lịch sử.')
    if st.button('Xóa lịch sử phiên'):st.session_state.history=[]
    hist=pd.DataFrame(st.session_state.history)
    if hist.empty:st.info('Chưa có dự đoán trong phiên này.')
    else:st.dataframe(hist,hide_index=True,width='stretch');download(hist,'lich_su_du_doan.csv')

else:
    st.subheader('Dữ liệu và phạm vi bài toán')
    st.write('AI4I 2020 có 10.000 mẫu mô phỏng. Chia phân tầng với seed 42: train 7.000, validation 1.500, test 1.500. Không dùng UDI, Product ID hoặc nhãn loại lỗi làm thuộc tính dự đoán.')
    st.link_button('Nguồn dữ liệu UCI','https://archive.ics.uci.edu/dataset/601/ai4i')
    st.dataframe(pd.DataFrame({'Cột':['Type']+NUM,'Ý nghĩa':['Loại sản phẩm L/M/H']+list(LABELS.values())}),hide_index=True,width='stretch')
    st.write('Sáu đầu vào được mã hóa thành bảy đặc trưng: năm biến số và hai cột Type_M, Type_H; loại L là nhóm tham chiếu. Nhiệt độ dùng Kelvin, không nhập trực tiếp độ C.')
    st.subheader('Ý nghĩa của kết quả')
    st.write('Nhãn 1 biểu thị mẫu được dự đoán hỏng, nhãn 0 biểu thị mẫu được dự đoán không hỏng. Nhóm dùng nhãn Machine failure trong dữ liệu để đánh giá kết quả.')
    st.write('Khi thông số tự nhập hoặc trong CSV vượt khoảng đã quan sát trên train, ứng dụng báo chưa đủ cơ sở kết luận. Phần kiểm chứng thực nghiệm đối chiếu toàn bộ tập test đã cố định với nhãn dữ liệu.')
    st.caption('Phạm vi nghiên cứu là phân loại trạng thái trên AI4I 2020. Thời điểm hỏng và tình trạng của máy thật cần dữ liệu kiểm chứng riêng.')

st.markdown('<div class="footer">NHÓM 16 · HỌC MÁY &nbsp; / &nbsp; AI4I 2020 · Demo nghiên cứu trên dữ liệu mô phỏng</div>',unsafe_allow_html=True)
