/*  ca_mcgill_sable_soot_baf_Baf.h -- from Java class ca.mcgill.sable.soot.baf.Baf  */
/*  created by Toba  */

#ifndef h_ca_mcgill_sable_soot_baf_Baf
#define h_ca_mcgill_sable_soot_baf_Baf

#define init_ca_mcgill_sable_soot_baf_Baf() if (cl_ca_mcgill_sable_soot_baf_Baf.C.needinit) initclass(&cl_ca_mcgill_sable_soot_baf_Baf.C)

Void	init__6GrRQ(Object);
Object	v__iF6BX(void);
Object	buildBodyOfFrom_SBi_j3ZqN(Object,Object,Object,Int);
Void	clinit__tMTqG(void);

struct mt_ca_mcgill_sable_soot_baf_Baf {
    struct {TobaMethodInvokeType itype;Void(*f)(Object);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} init__6GrRQ;
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
    struct {TobaMethodInvokeType itype;Object(*f)(Object,Object,Object,Int);
	const Char *name_chars;int name_len;const Char *sig_chars;int sig_len;
	int localp;int access;int classfilePos;Class *xlist;} buildBodyOfFrom_SBi_j3ZqN;
};

struct cv_ca_mcgill_sable_soot_baf_Baf {
    Object bafRepresentation;
};

extern struct cl_ca_mcgill_sable_soot_baf_Baf {
    struct class C;
    struct mt_ca_mcgill_sable_soot_baf_Baf M;
    struct cv_ca_mcgill_sable_soot_baf_Baf V;
} cl_ca_mcgill_sable_soot_baf_Baf;

struct in_ca_mcgill_sable_soot_baf_Baf {
    struct cl_ca_mcgill_sable_soot_baf_Baf *class;
    struct monitor *monitor;
};

#endif /* h_ca_mcgill_sable_soot_baf_Baf */
