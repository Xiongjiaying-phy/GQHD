# 2026-09-08 GQHD1–4 目标函数核查

这是七模型包中四组 GQHD 的拟合目标，不是后来五个参考模型重新拟合的目标。TM1、NL3、FSU-δ6.7 在本包中仅作参考，没有参与这次优化。

入口 run_gqhd_tidal_refine.evaluate -> run_gqhd_continuous_fullspace.evaluate + run_gqhd_pressure_refine.pressure_score + gqhd_gw170817_joint.GW170817Joint.log_integral。

总 loss = MR profile 项 + nuclear + stability + symmetry + thermo + mass + local_distance + flow_pressure + GW170817。源码包含所有系数；Summary.json 是四组实际分解。GQHD1.json 等包括每个 HIC 网格点的 n、P、上下界、带外距离及其平方，和用梯形积分复算的 HIC 项。

## HIC

压力是对称核物质 P=n(μn+μp)/2−ε，在 nn=np=n/2 求均场和化学势。不是 β 平衡压力。

固定 u=n/0.16，范围2–4.6，即 n=0.32–0.736 fm^-3。网格为53个等距点与上下边界全部顶点的并集。flow.json 保存原始顶点；在 u 上线性插值 ln P，不用各模型自身的饱和密度缩放。

r=max(ln(Plo/P),0)+max(ln(P/Phi),0)，再除以 ln(Phi/Plo)/2。
Lhic=10/(4.6−2) × integral r(u)^2 du，梯形积分。

带内严格为零，边界外才惩罚。不吸引到色带中心。weight=10 是人为超参数，不是来自实验的精确统计权重。该带由 Danielewicz、Lacey、Lynch (2002) Fig.3 手工数字化，不是作者原始数值表。Kaon、手征EFT后来加入展示，但本轮未进入目标函数。

## 核物质

六项平方偏差中心/尺度：(n0:0.155/0.012 fm^-3)，(E0:-16/2 MeV)，(K:240/80 MeV)，(J:31/4 MeV)，(L:60/30 MeV)，(M*/M:0.65/0.15)。此外加250倍带外平方屏障，范围/尺度依次：[.125,.185]/.01，[-20,-12]/2，[100,450]/80，[20,45]/5，[-30,160]/30，[.35,.95]/.15。它们是拟合惩罚尺度，不能自动解释成独立实验1σ。

## MR

observation_term 在稳定MR曲线上用181个质量点最小化 ((M−Mobs)/σM)^2+((R(M)−Robs)/σR)^2；上下侧采用各自误差，质量搜索限在观测±4σ与模型质量范围的交集。无交集记1e5。这是边缘误差独立近似的profile分数，不是图中公开二维后验的直接似然，也没有质量积分。

GQHD1:6620+4715；GQHD2:再加3329；GQHD3:再加347；GQHD4:四者全部。观测中心/上下误差原值见 run_gqhd_mr_constraint_fit.py 的 OBSERVATIONS。

## GW

GW项为 -2 log I，I是在 EOS 的 Λ(M) 轨道上对探测器系啁啾质量和质量比积分的四维 KDE 密度。带宽.30；使用释放的低自旋 PhenomPNRT 样本，红移.0099；采用与PE一致的质量先验约定。32×96 Gauss-Legendre网格，密集验证对比64×192，要求 |ΔlogI|≤.02。源码明确反射边界、质量和Λ范围。该项有任意公共加法常数，其绝对大小不能直接理解为拟合差的大小。不是应变重分析。

## 物理惩罚与接受门槛

以 [x]+ = max(x,0)：
stability = 3000[-λmin/20]+² + 12[(5−λmin)/5]+²；
symmetry = 3000[-Esym,min/5]+² + 10[(2−Esym,min)/2]+²；
thermo = 6000[-cs²min/.02]+² + 6000[(cs²max−1)/.05]+² + 6000[-ΔPmin/.2]+²；
mass = 1000[(2.02−Mmax)/.05]+²。
local_distance = .12 mean(((x−anchor)/max(.10|anchor|,.04 WIDTH))²)，WIDTH是各参数全局上限减下限（源码LOW/HIGH）。GQHD1锚为旧GQHD2，GQHD2–4锚为旧GQHD1，不能与交付的重新编号混淆。

硬检查与这些软惩罚并存：λmin>0，Esym,min>0，ΔPmin>0，0<cs²≤1，Mmax≥2，核物质在上述范围，正Maxwell密度/能量跳跃。检查在有限网格上进行，不是全密度解析证明。潮汐阶段未通过物理门槛的点最低记1e12；数值异常常记1e15。

## 优化过程和复现范围

21参数局部有界微调，尺度max(.05|x0|,.005 WIDTH)，边界x0±尺度且受全局边界约束。16随机试探、2个起点；归一化步长.02的中心有限差分，投影最速下降；线搜索步长.3/.1/.03/.01，每两轮密集验算。非全局最优证明。

这是原代码与评分审计包，不是独立可运行的完整环境：运行拟合还需原项目的EOS/TOV依赖、BPS、种子和GW样本。每项数值可以由所附JSON和公式核对。新增审计不改变原参数或结果。
