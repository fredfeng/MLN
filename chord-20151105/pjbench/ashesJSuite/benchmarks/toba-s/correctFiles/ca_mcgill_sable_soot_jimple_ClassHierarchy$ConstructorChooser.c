/*  ca_mcgill_sable_soot_jimple_ClassHierarchy$ConstructorChooser.c -- from Java class ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser  */
/*  created by Toba  */

#include "toba.h"
#include "ca_mcgill_sable_soot_jimple_ClassHierarchy$ConstructorChooser.h"
#include "ca_mcgill_sable_soot_TypeSwitch.h"
#include "ca_mcgill_sable_util_Switch.h"
#include "java_lang_Object.h"
#include "java_lang_String.h"
#include "java_lang_Class.h"
#include "ca_mcgill_sable_soot_Type.h"
#include "ca_mcgill_sable_soot_jimple_ClassHierarchy.h"
#include "ca_mcgill_sable_soot_jimple_ClassHierarchy$ToInt.h"
#include "ca_mcgill_sable_soot_jimple_ClassHierarchy$TypeNode.h"

static const Class supers[] = {
    &cl_ca_mcgill_sable_soot_jimp_OGkBH.C,
    &cl_ca_mcgill_sable_soot_TypeSwitch.C,
    &cl_java_lang_Object.C,
};

static const Class inters[] = {
    &cl_ca_mcgill_sable_util_Switch.C,
};

static const Class others[] = {
    &cl_ca_mcgill_sable_soot_Type.C,
    &cl_ca_mcgill_sable_soot_jimple_ClassHierarchy.C,
    &cl_ca_mcgill_sable_soot_jimp_fo53n.C,
    &cl_ca_mcgill_sable_soot_jimp_vtc6C.C,
};

extern const Char ch_ca_mcgill_sable_soot_jimp_OGkBH[];
extern const void *st_ca_mcgill_sable_soot_jimp_OGkBH[];
extern Class xt_init_C_femWZ[];
extern Class xt_typeNode_T_08GxW[];
extern Class xt_caseRefType_R_gZ5TV[];
extern Class xt_caseArrayType_A_UlMno[];
extern Class xt_defaultCase_T_BHmmF[];

#define HASHMASK 0x7
/*  1.  8942e761  (1)  hashCode  */
/*  2.  4c0d6fd2  (2)  clone  */
/*  6.  c2aafd4e  (6)  equals  */
/*  7.  489f8e6f  (7)  toString  */
static const struct ihash htable[9] = {
    0, 0,
    -1992104095, &cl_ca_mcgill_sable_soot_jimp_OGkBH.M.hashCode__8wJNW,
    1275949010, &cl_ca_mcgill_sable_soot_jimp_OGkBH.M.clone__dslwm, 0, 0,
    0, 0, 0, 0,
    -1028981426, &cl_ca_mcgill_sable_soot_jimp_OGkBH.M.equals_O_Ba6e0,
    1218416239, &cl_ca_mcgill_sable_soot_jimp_OGkBH.M.toString__4d9OF,
    0, 0,
};

static const CARRAY(61) nmchars = {&acl_char, 0, 61, 0,
'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','j','i','m','p','l','e','.','C','l','a','s','s','H','i','e',
'r','a','r','c','h','y','$','C','o','n','s','t','r','u','c','t','o','r',
'C','h','o','o','s','e','r'};
static struct in_java_lang_String classname =
    { &cl_java_lang_String, 0, (Object)&nmchars, 0, 61 };
static const Char nmiv_0[] = {
'r','e','s','u','l','t'};
static const Char sgiv_0[] = {
'L','j','a','v','a','/','l','a','n','g','/','O','b','j','e','c','t',';'};
static const Char nmiv_1[] = {
't','h','i','s','$','0'};
static const Char sgiv_1[] = {
'L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s',
'o','o','t','/','j','i','m','p','l','e','/','C','l','a','s','s','H','i',
'e','r','a','r','c','h','y',';'};
static const Char nmiv_2[] = {
'r','e','s','u','l','t'};
static const Char sgiv_2[] = {
'L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s',
'o','o','t','/','j','i','m','p','l','e','/','C','l','a','s','s','H','i',
'e','r','a','r','c','h','y','$','T','y','p','e','N','o','d','e',';'};
static const Char nmim_0[] = {
'<','i','n','i','t','>'};
static const Char sgim_0[] = {
'(',')','V'};
static const Char nmim_1[] = {
'c','l','o','n','e'};
static const Char sgim_1[] = {
'(',')','L','j','a','v','a','/','l','a','n','g','/','O','b','j','e','c',
't',';'};
static const Char nmim_2[] = {
'e','q','u','a','l','s'};
static const Char sgim_2[] = {
'(','L','j','a','v','a','/','l','a','n','g','/','O','b','j','e','c','t',
';',')','Z'};
static const Char nmim_3[] = {
'f','i','n','a','l','i','z','e'};
static const Char sgim_3[] = {
'(',')','V'};
static const Char nmim_4[] = {
'g','e','t','C','l','a','s','s'};
static const Char sgim_4[] = {
'(',')','L','j','a','v','a','/','l','a','n','g','/','C','l','a','s','s',
';'};
static const Char nmim_5[] = {
'h','a','s','h','C','o','d','e'};
static const Char sgim_5[] = {
'(',')','I'};
static const Char nmim_6[] = {
'n','o','t','i','f','y'};
static const Char sgim_6[] = {
'(',')','V'};
static const Char nmim_7[] = {
'n','o','t','i','f','y','A','l','l'};
static const Char sgim_7[] = {
'(',')','V'};
static const Char nmim_8[] = {
't','o','S','t','r','i','n','g'};
static const Char sgim_8[] = {
'(',')','L','j','a','v','a','/','l','a','n','g','/','S','t','r','i','n',
'g',';'};
static const Char nmim_9[] = {
'w','a','i','t'};
static const Char sgim_9[] = {
'(',')','V'};
static const Char nmim_10[] = {
'w','a','i','t'};
static const Char sgim_10[] = {
'(','J',')','V'};
static const Char nmim_11[] = {
'w','a','i','t'};
static const Char sgim_11[] = {
'(','J','I',')','V'};
static const Char nmim_12[] = {
'c','a','s','e','A','r','r','a','y','T','y','p','e'};
static const Char sgim_12[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','A','r','r','a','y','T','y','p','e',';',')','V'};
static const Char nmim_13[] = {
'c','a','s','e','B','o','o','l','e','a','n','T','y','p','e'};
static const Char sgim_13[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','B','o','o','l','e','a','n','T','y','p','e',';',')',
'V'};
static const Char nmim_14[] = {
'c','a','s','e','B','y','t','e','T','y','p','e'};
static const Char sgim_14[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','B','y','t','e','T','y','p','e',';',')','V'};
static const Char nmim_15[] = {
'c','a','s','e','C','h','a','r','T','y','p','e'};
static const Char sgim_15[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','C','h','a','r','T','y','p','e',';',')','V'};
static const Char nmim_16[] = {
'c','a','s','e','D','o','u','b','l','e','T','y','p','e'};
static const Char sgim_16[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','D','o','u','b','l','e','T','y','p','e',';',')','V'};
static const Char nmim_17[] = {
'c','a','s','e','F','l','o','a','t','T','y','p','e'};
static const Char sgim_17[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','F','l','o','a','t','T','y','p','e',';',')','V'};
static const Char nmim_18[] = {
'c','a','s','e','I','n','t','T','y','p','e'};
static const Char sgim_18[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','I','n','t','T','y','p','e',';',')','V'};
static const Char nmim_19[] = {
'c','a','s','e','L','o','n','g','T','y','p','e'};
static const Char sgim_19[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','L','o','n','g','T','y','p','e',';',')','V'};
static const Char nmim_20[] = {
'c','a','s','e','R','e','f','T','y','p','e'};
static const Char sgim_20[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','R','e','f','T','y','p','e',';',')','V'};
static const Char nmim_21[] = {
'c','a','s','e','S','h','o','r','t','T','y','p','e'};
static const Char sgim_21[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','S','h','o','r','t','T','y','p','e',';',')','V'};
static const Char nmim_22[] = {
'c','a','s','e','S','t','m','t','A','d','d','r','e','s','s','T','y','p',
'e'};
static const Char sgim_22[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','S','t','m','t','A','d','d','r','e','s','s','T','y',
'p','e',';',')','V'};
static const Char nmim_23[] = {
'c','a','s','e','U','n','k','n','o','w','n','T','y','p','e'};
static const Char sgim_23[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','U','n','k','n','o','w','n','T','y','p','e',';',')',
'V'};
static const Char nmim_24[] = {
'c','a','s','e','V','o','i','d','T','y','p','e'};
static const Char sgim_24[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','V','o','i','d','T','y','p','e',';',')','V'};
static const Char nmim_25[] = {
'c','a','s','e','N','u','l','l','T','y','p','e'};
static const Char sgim_25[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','N','u','l','l','T','y','p','e',';',')','V'};
static const Char nmim_26[] = {
'c','a','s','e','E','r','r','o','n','e','o','u','s','T','y','p','e'};
static const Char sgim_26[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','E','r','r','o','n','e','o','u','s','T','y','p','e',
';',')','V'};
static const Char nmim_27[] = {
'd','e','f','a','u','l','t','C','a','s','e'};
static const Char sgim_27[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','T','y','p','e',';',')','V'};
static const Char nmim_28[] = {
'c','a','s','e','D','e','f','a','u','l','t'};
static const Char sgim_28[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','T','y','p','e',';',')','V'};
static const Char nmim_29[] = {
's','e','t','R','e','s','u','l','t'};
static const Char sgim_29[] = {
'(','L','j','a','v','a','/','l','a','n','g','/','O','b','j','e','c','t',
';',')','V'};
static const Char nmim_30[] = {
'g','e','t','R','e','s','u','l','t'};
static const Char sgim_30[] = {
'(',')','L','j','a','v','a','/','l','a','n','g','/','O','b','j','e','c',
't',';'};
static const Char nmim_31[] = {
'<','i','n','i','t','>'};
static const Char sgim_31[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','j','i','m','p','l','e','/','C','l','a','s','s','H',
'i','e','r','a','r','c','h','y',';',')','V'};
static const Char nmim_32[] = {
't','y','p','e','N','o','d','e'};
static const Char sgim_32[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','T','y','p','e',';',')','L','c','a','/','m','c','g',
'i','l','l','/','s','a','b','l','e','/','s','o','o','t','/','j','i','m',
'p','l','e','/','C','l','a','s','s','H','i','e','r','a','r','c','h','y',
'$','T','y','p','e','N','o','d','e',';'};

static struct vt_generic cv_table[] = {
    {0}
};

#ifndef offsetof
#define offsetof(s,m) ((int)&(((s *)0))->m)
#endif
static struct vt_generic iv_table[] = {
    { offsetof(struct in_ca_mcgill_sable_soot_jimp_OGkBH, result), 0,(const Char *)&nmiv_0,6,(const Char *)&sgiv_0,18,0,0x0,0}, 
    { offsetof(struct in_ca_mcgill_sable_soot_jimp_OGkBH, this0_oYxUV), 0,(const Char *)&nmiv_1,6,(const Char *)&sgiv_1,44,1,0x12,0}, 
    { offsetof(struct in_ca_mcgill_sable_soot_jimp_OGkBH, result), 0,(const Char *)&nmiv_2,6,(const Char *)&sgiv_2,53,1,0x2,1}, 
};
#undef offsetof

static struct mt_generic sm_table[] = {
    {TMIT_undefined}
};

#ifndef h_toba_classfile_ClassRef
extern struct cl_generic cl_toba_classfile_ClassRef;
#endif /* h_toba_classfile_ClassRef */
static struct { /* pseudo in_toba_classfile_ClassRef */
   struct cl_generic *class;
   struct monitor *monitor;
   Object name;
   Object refClass;
   Object loadedRefdClasses;
} inr_ca_mcgill_sable_soot_jimp_OGkBH = {
  (struct cl_generic *)&cl_toba_classfile_ClassRef.C, 0, &classname, &cl_ca_mcgill_sable_soot_jimp_OGkBH.C.classclass, 0};

struct cl_ca_mcgill_sable_soot_jimp_OGkBH cl_ca_mcgill_sable_soot_jimp_OGkBH = { {
    1, 0,
    &classname,
    &cl_java_lang_Class.C, 0,
    sizeof(struct in_ca_mcgill_sable_soot_jimp_OGkBH),
    33,
    0,
    3,
    0,
    3, supers,
    1, 0, inters, HASHMASK, htable,
    4, others,
    0, 0,
    ch_ca_mcgill_sable_soot_jimp_OGkBH,
    st_ca_mcgill_sable_soot_jimp_OGkBH,
    0,
    throwNoSuchMethodError,
    finalize__UKxhs,
    0,
    0,
    43,
    0x20,
    0,
    (struct in_toba_classfile_ClassRef *)&inr_ca_mcgill_sable_soot_jimp_OGkBH,
    iv_table, cv_table,
    sm_table},
    { /* methodsigs */
	{TMIT_native_code, init__hEn3f,(const Char *)&nmim_0,6,
	(const Char *)&sgim_0,3,0,0x1,19,0},
	{TMIT_native_code, clone__dslwm,(const Char *)&nmim_1,5,
	(const Char *)&sgim_1,20,0,0x8104,2,0},
	{TMIT_native_code, equals_O_Ba6e0,(const Char *)&nmim_2,6,
	(const Char *)&sgim_2,21,0,0x8001,3,0},
	{TMIT_native_code, finalize__UKxhs,(const Char *)&nmim_3,8,
	(const Char *)&sgim_3,3,0,0x4,4,0},
	{TMIT_native_code, getClass__zh19H,(const Char *)&nmim_4,8,
	(const Char *)&sgim_4,19,0,0x111,5,0},
	{TMIT_native_code, hashCode__8wJNW,(const Char *)&nmim_5,8,
	(const Char *)&sgim_5,3,0,0x8101,6,0},
	{TMIT_native_code, notify__ne4bk,(const Char *)&nmim_6,6,
	(const Char *)&sgim_6,3,0,0x111,7,0},
	{TMIT_native_code, notifyAll__iTnlk,(const Char *)&nmim_7,9,
	(const Char *)&sgim_7,3,0,0x111,8,0},
	{TMIT_native_code, toString__4d9OF,(const Char *)&nmim_8,8,
	(const Char *)&sgim_8,20,0,0x8001,10,0},
	{TMIT_native_code, wait__Zlr2b,(const Char *)&nmim_9,4,
	(const Char *)&sgim_9,3,0,0x11,11,0},
	{TMIT_native_code, wait_l_1Iito,(const Char *)&nmim_10,4,
	(const Char *)&sgim_10,4,0,0x111,12,0},
	{TMIT_native_code, wait_li_07Ea2,(const Char *)&nmim_11,4,
	(const Char *)&sgim_11,5,0,0x11,13,0},
	{TMIT_native_code, caseArrayType_A_UlMno,(const Char *)&nmim_12,13,
	(const Char *)&sgim_12,35,1,0x1,3,xt_caseArrayType_A_UlMno},
	{TMIT_native_code, caseBooleanType_B_OUFIT,(const Char *)&nmim_13,15,
	(const Char *)&sgim_13,37,0,0x1,1,0},
	{TMIT_native_code, caseByteType_B_WcDiD,(const Char *)&nmim_14,12,
	(const Char *)&sgim_14,34,0,0x1,2,0},
	{TMIT_native_code, caseCharType_C_8951U,(const Char *)&nmim_15,12,
	(const Char *)&sgim_15,34,0,0x1,3,0},
	{TMIT_native_code, caseDoubleType_D_uhKoo,(const Char *)&nmim_16,14,
	(const Char *)&sgim_16,36,0,0x1,4,0},
	{TMIT_native_code, caseFloatType_F_QRtel,(const Char *)&nmim_17,13,
	(const Char *)&sgim_17,35,0,0x1,5,0},
	{TMIT_native_code, caseIntType_I_ODuHy,(const Char *)&nmim_18,11,
	(const Char *)&sgim_18,33,0,0x1,6,0},
	{TMIT_native_code, caseLongType_L_oZuIo,(const Char *)&nmim_19,12,
	(const Char *)&sgim_19,34,0,0x1,7,0},
	{TMIT_native_code, caseRefType_R_gZ5TV,(const Char *)&nmim_20,11,
	(const Char *)&sgim_20,33,1,0x1,2,xt_caseRefType_R_gZ5TV},
	{TMIT_native_code, caseShortType_S_2MJrs,(const Char *)&nmim_21,13,
	(const Char *)&sgim_21,35,0,0x1,9,0},
	{TMIT_native_code, caseStmtAddressType_S_6w2qx,(const Char *)&nmim_22,19,
	(const Char *)&sgim_22,41,0,0x1,10,0},
	{TMIT_native_code, caseUnknownType_U_Mn0wN,(const Char *)&nmim_23,15,
	(const Char *)&sgim_23,37,0,0x1,11,0},
	{TMIT_native_code, caseVoidType_V_287iS,(const Char *)&nmim_24,12,
	(const Char *)&sgim_24,34,0,0x1,12,0},
	{TMIT_native_code, caseNullType_N_ipySa,(const Char *)&nmim_25,12,
	(const Char *)&sgim_25,34,0,0x1,13,0},
	{TMIT_native_code, caseErroneousType_E_Qb6aF,(const Char *)&nmim_26,17,
	(const Char *)&sgim_26,39,0,0x1,14,0},
	{TMIT_native_code, defaultCase_T_BHmmF,(const Char *)&nmim_27,11,
	(const Char *)&sgim_27,30,1,0x1,4,xt_defaultCase_T_BHmmF},
	{TMIT_native_code, caseDefault_T_Hrrk6,(const Char *)&nmim_28,11,
	(const Char *)&sgim_28,30,0,0x1,16,0},
	{TMIT_native_code, setResult_O_mipzV,(const Char *)&nmim_29,9,
	(const Char *)&sgim_29,21,0,0x1,17,0},
	{TMIT_native_code, getResult__q3qpX,(const Char *)&nmim_30,9,
	(const Char *)&sgim_30,20,0,0x1,18,0},
	{TMIT_native_code, init_C_femWZ,(const Char *)&nmim_31,6,
	(const Char *)&sgim_31,47,1,0x0,0,xt_init_C_femWZ},
	{TMIT_native_code, typeNode_T_08GxW,(const Char *)&nmim_32,8,
	(const Char *)&sgim_32,82,1,0x0,1,xt_typeNode_T_08GxW},
    } /* end of methodsigs */
};


/*M init_C_femWZ: ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser.<init>(Lca/mcgill/sable/soot/jimple/ClassHierarchy;)V */

Class xt_init_C_femWZ[] = { 0 };

Void init_C_femWZ(Object p0, Object p1)
{
Object a0, a1, a2;
Object av0, av1;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	init__hEn3f(a1);
	a1 = av0;
	a2 = av1;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->this0_oYxUV = a2;
	a1 = av0;
	a2 = av1;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->this0_oYxUV = a2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M typeNode_T_08GxW: ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser.typeNode(Lca/mcgill/sable/soot/Type;)Lca/mcgill/sable/soot/jimple/ClassHierarchy$TypeNode; */

Class xt_typeNode_T_08GxW[] = { 0 };

Object typeNode_T_08GxW(Object p0, Object p1)
{
Object a0, a1, a2;
Object av0, av1;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av1;
	a2 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_Type*)a1)->class->M.
		apply_S_1raGs.f(a1,a2);
	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	a1 = ((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->result;
	return a1;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M caseRefType_R_gZ5TV: ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser.caseRefType(Lca/mcgill/sable/soot/RefType;)V */

Class xt_caseRefType_R_gZ5TV[] = { 0 };

Void caseRefType_R_gZ5TV(Object p0, Object p1)
{
Object a0, a1, a2, a3, a4, a5;
Object av0, av1;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av0;
	a2 = new(&cl_ca_mcgill_sable_soot_jimp_vtc6C.C);
	a3 = a2;
	a4 = av0;
	if (!a4) { 
		SetNPESource(); goto NULLX;
	}
	a4 = ((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a4)->this0_oYxUV;
	a5 = av1;
	if (!a3) { 
		SetNPESource(); goto NULLX;
	}
	init_CR_sUhxT(a3,a4,a5);
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->result = a2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M caseArrayType_A_UlMno: ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser.caseArrayType(Lca/mcgill/sable/soot/ArrayType;)V */

Class xt_caseArrayType_A_UlMno[] = { 0 };

Void caseArrayType_A_UlMno(Object p0, Object p1)
{
Object a0, a1, a2, a3, a4, a5;
Object av0, av1;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av0;
	a2 = new(&cl_ca_mcgill_sable_soot_jimp_vtc6C.C);
	a3 = a2;
	a4 = av0;
	if (!a4) { 
		SetNPESource(); goto NULLX;
	}
	a4 = ((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a4)->this0_oYxUV;
	a5 = av1;
	if (!a3) { 
		SetNPESource(); goto NULLX;
	}
	init_CA_eQXEg(a3,a4,a5);
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->result = a2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M defaultCase_T_BHmmF: ca.mcgill.sable.soot.jimple.ClassHierarchy$ConstructorChooser.defaultCase(Lca/mcgill/sable/soot/Type;)V */

Class xt_defaultCase_T_BHmmF[] = { 0 };

Void defaultCase_T_BHmmF(Object p0, Object p1)
{
Object a0, a1, a2, a3, a4, a5;
Object av0, av1;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av0;
	a2 = new(&cl_ca_mcgill_sable_soot_jimp_vtc6C.C);
	a3 = a2;
	a4 = av0;
	if (!a4) { 
		SetNPESource(); goto NULLX;
	}
	a4 = ((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a4)->this0_oYxUV;
	a5 = av1;
	if (!a3) { 
		SetNPESource(); goto NULLX;
	}
	init_CT_FfN7l(a3,a4,a5);
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_jimp_OGkBH*)a1)->result = a2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}



const Char ch_ca_mcgill_sable_soot_jimp_OGkBH[] = {  /* string pool */'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','j','i','m','p','l','e','.','C','l','a','s','s','H','i','e',
'r','a','r','c','h','y','$','C','o','n','s','t','r','u','c','t','o','r',
'C','h','o','o','s','e','r'};

const void *st_ca_mcgill_sable_soot_jimp_OGkBH[] = {
    ch_ca_mcgill_sable_soot_jimp_OGkBH+61,	/* 0. ca.mcgill.sable.soot.jimple.ClassHierarc */
    0};
