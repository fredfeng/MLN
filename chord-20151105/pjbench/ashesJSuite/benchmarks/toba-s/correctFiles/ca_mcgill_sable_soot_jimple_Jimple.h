/*  ca_mcgill_sable_soot_jimple_Jimple.h -- from Java class ca.mcgill.sable.soot.jimple.Jimple  */
/*  created by Toba  */

#ifndef h_ca_mcgill_sable_soot_jimple_Jimple
#define h_ca_mcgill_sable_soot_jimple_Jimple

#define init_ca_mcgill_sable_soot_jimple_Jimple() if (cl_ca_mcgill_sable_soot_jimple_Jimple.C.needinit) initclass(&cl_ca_mcgill_sable_soot_jimple_Jimple.C)

Void	init__MTb7e(Object);
Object	v__WbL8F(void);
Object	newBody_S_inFod(Object,Object);
Object	buildBodyOfFrom_SBi_LbOSp(Object,Object,Object,Int);
Object	newXorExpr_VV_FdTYB(Object,Object,Object);
Object	newUshrExpr_VV_nbYv9(Object,Object,Object);
Object	newSubExpr_VV_t6rHK(Object,Object,Object);
Object	newShrExpr_VV_7USr6(Object,Object,Object);
Object	newShlExpr_VV_Z3Hxs(Object,Object,Object);
Object	newRemExpr_VV_PVebu(Object,Object,Object);
Object	newOrExpr_VV_Do5lF(Object,Object,Object);
Object	newNeExpr_VV_1FBI4(Object,Object,Object);
Object	newMulExpr_VV_jXO1w(Object,Object,Object);
Object	newLeExpr_VV_f80QC(Object,Object,Object);
Object	newGeExpr_VV_f7ftO(Object,Object,Object);
Object	newEqExpr_VV_rnLgP(Object,Object,Object);
Object	newDivExpr_VV_5pVe7(Object,Object,Object);
Object	newCmplExpr_VV_ZqHbU(Object,Object,Object);
Object	newCmpgExpr_VV_VvlhW(Object,Object,Object);
Object	newCmpExpr_VV_r3FPU(Object,Object,Object);
Object	newGtExpr_VV_Vfhui(Object,Object,Object);
Object	newLtExpr_VV_Vg2R6(Object,Object,Object);
Object	newAddExpr_VV_d9P1D(Object,Object,Object);
Object	newAndExpr_VV_vu8Jd(Object,Object,Object);
Object	newNegExpr_V_vFo9X(Object,Object);
Object	newLengthExpr_V_TbHgT(Object,Object);
Object	newCastExpr_VT_AM1Sk(Object,Object,Object);
Object	newInstanceOfExpr_VT_4PuH5(Object,Object,Object);
Object	newNewExpr_R_zYk3f(Object,Object);
Object	newNewArrayExpr_TV_y7F0E(Object,Object,Object);
Object	newNewMultiArrayExpr_AL_tuLqx(Object,Object,Object);
Object	newStaticInvokeExpr_SL_uU9rl(Object,Object,Object);
Object	newSpecialInvokeExpr_LSL_4bbUn(Object,Object,Object,Object);
Object	newVirtualInvokeExpr_LSL_OgZSl(Object,Object,Object,Object);
Object	newInterfaceInvokeExp_LSL_83yBW(Object,Object,Object,Object);
Object	newThrowStmt_V_f0VDI(Object,Object);
Object	newExitMonitorStmt_V_FDF1h(Object,Object);
Object	newEnterMonitorStmt_V_5LDIC(Object,Object);
Object	newBreakpointStmt__BYvBn(Object);
Object	newGotoStmt_U_qRelW(Object,Object);
Object	newNopStmt__PVmrT(Object);
Object	newReturnVoidStmt__bav3J(Object);
Object	newReturnStmt_V_7EEj7(Object,Object);
Object	newRetStmt_V_Hjahj(Object,Object);
Object	newIfStmt_VU_G1XNZ(Object,Object,Object);
Object	newIdentityStmt_VV_veFeD(Object,Object,Object);
Object	newAssignStmt_VV_VK6ax(Object,Object,Object);
Object	newInvokeStmt_V_Por4z(Object,Object);
Object	newTableSwitchStmt_ViiLU_urD3h(Object,Object,Int,Int,Object,Object);
Object	newLookupSwitchStmt_VLLU_8Z2Wo(Object,Object,Object,Object,Object);
Object	newLocal_ST_UrNI5(Object,Object,Object);
Object	newTrap_SUUU_iT5VH(Object,Object,Object,Object,Object);
Object	newStaticFieldRef_S_NA8MK(Object,Object);
Object	newThisRef_S_7k96k(Object,Object);
Object	newParameterRef_Si_1GJaU(Object,Object,Int);
Object	newInstanceFieldRef_VS_lXqxs(Object,Object,Object);
Object	newCaughtExceptionRef_J_NQePj(Object,Object);
Object	newArrayRef_VV_vy9KY(Object,Object,Object);
Object	newVariableBox_V_sAbI7(Object,Object);
Object	newLocalBox_V_BsxZy(Object,Object);
Object	newRValueBox_V_fJ2TY(Object,Object);
Object	newImmediateBox_V_7jLB8(Object,Object);
Object	newArgBox_V_G9BVX(Object,Object);
Object	newIdentityRefBox_V_FOSWQ(Object,Object);
Object	newConditionExprBox_V_KdLXR(Object,Object);
Object	newInvokeExprBox_V_TXEdu(Object,Object);
Object	newStmtBox_U_e3goW(Object,Object);
Void	clinit__FcYlj(void);

struct mt_ca_mcgill_sable_soot_jimple_Jimple {
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} init__MTb7e;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} clone__dslwm;
    struct {TobaMethodInvokeType itype;Boolean(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} equals_O_Ba6e0;
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} finalize__UKxhs;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} getClass__zh19H;
    struct {TobaMethodInvokeType itype;Int(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} hashCode__8wJNW;
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} notify__ne4bk;
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} notifyAll__iTnlk;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} toString__4d9OF;
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} wait__Zlr2b;
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Long);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} wait_l_1Iito;
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Long,Int);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} wait_li_07Ea2;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newBody_S_inFod;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Int);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} buildBodyOfFrom_SBi_LbOSp;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newXorExpr_VV_FdTYB;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newUshrExpr_VV_nbYv9;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newSubExpr_VV_t6rHK;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newShrExpr_VV_7USr6;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newShlExpr_VV_Z3Hxs;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newRemExpr_VV_PVebu;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newOrExpr_VV_Do5lF;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNeExpr_VV_1FBI4;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newMulExpr_VV_jXO1w;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLeExpr_VV_f80QC;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newGeExpr_VV_f7ftO;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newEqExpr_VV_rnLgP;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newDivExpr_VV_5pVe7;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newCmplExpr_VV_ZqHbU;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newCmpgExpr_VV_VvlhW;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newCmpExpr_VV_r3FPU;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newGtExpr_VV_Vfhui;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLtExpr_VV_Vg2R6;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newAddExpr_VV_d9P1D;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newAndExpr_VV_vu8Jd;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNegExpr_V_vFo9X;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLengthExpr_V_TbHgT;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newCastExpr_VT_AM1Sk;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInstanceOfExpr_VT_4PuH5;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNewExpr_R_zYk3f;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNewArrayExpr_TV_y7F0E;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNewMultiArrayExpr_AL_tuLqx;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newStaticInvokeExpr_SL_uU9rl;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newSpecialInvokeExpr_LSL_4bbUn;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newVirtualInvokeExpr_LSL_OgZSl;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInterfaceInvokeExp_LSL_83yBW;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newThrowStmt_V_f0VDI;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newExitMonitorStmt_V_FDF1h;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newEnterMonitorStmt_V_5LDIC;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newBreakpointStmt__BYvBn;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newGotoStmt_U_qRelW;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newNopStmt__PVmrT;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newReturnVoidStmt__bav3J;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newReturnStmt_V_7EEj7;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newRetStmt_V_Hjahj;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newIfStmt_VU_G1XNZ;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newIdentityStmt_VV_veFeD;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newAssignStmt_VV_VK6ax;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInvokeStmt_V_Por4z;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Int,Int,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newTableSwitchStmt_ViiLU_urD3h;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLookupSwitchStmt_VLLU_8Z2Wo;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLocal_ST_UrNI5;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newTrap_SUUU_iT5VH;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newStaticFieldRef_S_NA8MK;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newThisRef_S_7k96k;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Int);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newParameterRef_Si_1GJaU;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInstanceFieldRef_VS_lXqxs;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newCaughtExceptionRef_J_NQePj;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newArrayRef_VV_vy9KY;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newVariableBox_V_sAbI7;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newLocalBox_V_BsxZy;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newRValueBox_V_fJ2TY;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newImmediateBox_V_7jLB8;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newArgBox_V_G9BVX;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newIdentityRefBox_V_FOSWQ;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newConditionExprBox_V_KdLXR;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInvokeExprBox_V_TXEdu;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newStmtBox_U_e3goW;
};

struct cv_ca_mcgill_sable_soot_jimple_Jimple {
    Object jimpleRepresentation;
};

extern struct cl_ca_mcgill_sable_soot_jimple_Jimple {
    struct class C;
    struct mt_ca_mcgill_sable_soot_jimple_Jimple M;
    struct cv_ca_mcgill_sable_soot_jimple_Jimple V;
} cl_ca_mcgill_sable_soot_jimple_Jimple;

struct in_ca_mcgill_sable_soot_jimple_Jimple {
    struct cl_ca_mcgill_sable_soot_jimple_Jimple *class;
    struct monitor *monitor;
};

#endif /* h_ca_mcgill_sable_soot_jimple_Jimple */