/*  ca_mcgill_sable_soot_jimple_FlowAnalysis.h -- from Java class ca.mcgill.sable.soot.jimple.FlowAnalysis  */
/*  created by Toba  */

#ifndef h_ca_mcgill_sable_soot_jimple_FlowAnalysis
#define h_ca_mcgill_sable_soot_jimple_FlowAnalysis

#define init_ca_mcgill_sable_soot_jimple_FlowAnalysis() (void)0

Void	init_S_yU0ut(Object,Object);
Object	newInitialFlow__V45Sv(Object);
Boolean	isForward__2joHv(Object);
Void	flowThrough_OSO_pPHbY(Object,Object,Object,Object);
Void	merge_OOO_LGa6A(Object,Object,Object,Object);
Void	copy_OO_u6KNw(Object,Object,Object);
Void	doAnalysis__orbB4(Object);
Object	getFlowAfterStmt_S_I5bTv(Object,Object);
Object	getFlowBeforeStmt_S_vm3Xd(Object,Object);

struct mt_ca_mcgill_sable_soot_jimple_FlowAnalysis {
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} init__AAyKx;
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
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} init_S_yU0ut;
    struct {TobaMethodInvokeType itype;Object(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} newInitialFlow__V45Sv;
    struct {TobaMethodInvokeType itype;Boolean(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} isForward__2joHv;
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} flowThrough_OSO_pPHbY;
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} merge_OOO_LGa6A;
    struct {TobaMethodInvokeType itype;Void(*f)(Object,Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} copy_OO_u6KNw;
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} doAnalysis__orbB4;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} getFlowAfterStmt_S_I5bTv;
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} getFlowBeforeStmt_S_vm3Xd;
};

extern struct cl_ca_mcgill_sable_soot_jimple_FlowAnalysis {
    struct class C;
    struct mt_ca_mcgill_sable_soot_jimple_FlowAnalysis M;
} cl_ca_mcgill_sable_soot_jimple_FlowAnalysis;

struct in_ca_mcgill_sable_soot_jimple_FlowAnalysis {
    struct cl_ca_mcgill_sable_soot_jimple_FlowAnalysis *class;
    struct monitor *monitor;
    Object stmtToAfterFlow;
    Object stmtToBeforeFlow;
    Object graph;
};

#endif /* h_ca_mcgill_sable_soot_jimple_FlowAnalysis */
