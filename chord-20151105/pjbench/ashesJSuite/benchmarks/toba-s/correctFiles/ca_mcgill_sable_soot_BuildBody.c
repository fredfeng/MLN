/*  ca_mcgill_sable_soot_BuildBody.c -- from Java class ca.mcgill.sable.soot.BuildBody  */
/*  created by Toba  */

#include "toba.h"
#include "ca_mcgill_sable_soot_BuildBody.h"
#include "ca_mcgill_sable_soot_BodyExpr.h"
#include "java_lang_Object.h"
#include "java_lang_String.h"
#include "java_lang_Class.h"
#include "ca_mcgill_sable_soot_BodyRepresentation.h"

static const Class supers[] = {
    &cl_ca_mcgill_sable_soot_BuildBody.C,
    &cl_java_lang_Object.C,
};

static const Class inters[] = {
    &cl_ca_mcgill_sable_soot_BodyExpr.C,
};

static const Class others[] = {
    &cl_ca_mcgill_sable_soot_BodyRepresentation.C,
};

extern const Char ch_ca_mcgill_sable_soot_BuildBody[];
extern const void *st_ca_mcgill_sable_soot_BuildBody[];
extern Class xt_init_BB_1pYXe[];
extern Class xt_init_BBi_Au8Qd[];
extern Class xt_resolveFor_S_ap4uc[];

#define HASHMASK 0xf
/*  1.  8942e761  (1)  hashCode  */
/*  2.  4c0d6fd2  (2)  clone  */
/*  6.  81cb61a6  (6)  resolveFor  */
/*  e.  c2aafd4e  (e)  equals  */
/*  f.  489f8e6f  (f)  toString  */
static const struct ihash htable[17] = {
    0, 0,
    -1992104095, &cl_ca_mcgill_sable_soot_BuildBody.M.hashCode__8wJNW,
    1275949010, &cl_ca_mcgill_sable_soot_BuildBody.M.clone__dslwm, 0, 0,
    0, 0, 0, 0,
    -2117377626, &cl_ca_mcgill_sable_soot_BuildBody.M.resolveFor_S_ap4uc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1028981426, &cl_ca_mcgill_sable_soot_BuildBody.M.equals_O_Ba6e0,
    1218416239, &cl_ca_mcgill_sable_soot_BuildBody.M.toString__4d9OF, 0, 0,
};

static const CARRAY(30) nmchars = {&acl_char, 0, 30, 0,
'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','B','u','i','l','d','B','o','d','y'};
static struct in_java_lang_String classname =
    { &cl_java_lang_String, 0, (Object)&nmchars, 0, 30 };
static const Char nmiv_0[] = {
'b','o','d','y','R','e','p'};
static const Char sgiv_0[] = {
'L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s',
'o','o','t','/','B','o','d','y','R','e','p','r','e','s','e','n','t','a',
't','i','o','n',';'};
static const Char nmiv_1[] = {
's','o','u','r','c','e','B','u','i','l','d','e','r'};
static const Char sgiv_1[] = {
'L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s',
'o','o','t','/','B','o','d','y','E','x','p','r',';'};
static const Char nmiv_2[] = {
'b','u','i','l','d','B','o','d','y','O','p','t','i','o','n','s'};
static const Char sgiv_2[] = {
'I'};
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
'<','i','n','i','t','>'};
static const Char sgim_12[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','B','o','d','y','R','e','p','r','e','s','e','n','t',
'a','t','i','o','n',';','L','c','a','/','m','c','g','i','l','l','/','s',
'a','b','l','e','/','s','o','o','t','/','B','o','d','y','E','x','p','r',
';',')','V'};
static const Char nmim_13[] = {
'<','i','n','i','t','>'};
static const Char sgim_13[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','B','o','d','y','R','e','p','r','e','s','e','n','t',
'a','t','i','o','n',';','L','c','a','/','m','c','g','i','l','l','/','s',
'a','b','l','e','/','s','o','o','t','/','B','o','d','y','E','x','p','r',
';','I',')','V'};
static const Char nmim_14[] = {
'r','e','s','o','l','v','e','F','o','r'};
static const Char sgim_14[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','S','o','o','t','M','e','t','h','o','d',';',')','L',
'c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s','o',
'o','t','/','B','o','d','y',';'};

static struct vt_generic cv_table[] = {
    {0}
};

#ifndef offsetof
#define offsetof(s,m) ((int)&(((s *)0))->m)
#endif
static struct vt_generic iv_table[] = {
    { offsetof(struct in_ca_mcgill_sable_soot_BuildBody, bodyRep), 0,(const Char *)&nmiv_0,7,(const Char *)&sgiv_0,41,1,0x11,0}, 
    { offsetof(struct in_ca_mcgill_sable_soot_BuildBody, sourceBuilder), 0,(const Char *)&nmiv_1,13,(const Char *)&sgiv_1,31,1,0x11,1}, 
    { offsetof(struct in_ca_mcgill_sable_soot_BuildBody, buildBodyOptions), 0,(const Char *)&nmiv_2,16,(const Char *)&sgiv_2,1,1,0x11,2}, 
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
} inr_ca_mcgill_sable_soot_BuildBody = {
  (struct cl_generic *)&cl_toba_classfile_ClassRef.C, 0, &classname, &cl_ca_mcgill_sable_soot_BuildBody.C.classclass, 0};

struct cl_ca_mcgill_sable_soot_BuildBody cl_ca_mcgill_sable_soot_BuildBody = { {
    1, 0,
    &classname,
    &cl_java_lang_Class.C, 0,
    sizeof(struct in_ca_mcgill_sable_soot_BuildBody),
    15,
    0,
    3,
    0,
    2, supers,
    1, 1, inters, HASHMASK, htable,
    1, others,
    0, 0,
    ch_ca_mcgill_sable_soot_BuildBody,
    st_ca_mcgill_sable_soot_BuildBody,
    0,
    throwNoSuchMethodError,
    finalize__UKxhs,
    0,
    0,
    43,
    0x21,
    0,
    (struct in_toba_classfile_ClassRef *)&inr_ca_mcgill_sable_soot_BuildBody,
    iv_table, cv_table,
    sm_table},
    { /* methodsigs */
	{TMIT_native_code, init__AAyKx,(const Char *)&nmim_0,6,
	(const Char *)&sgim_0,3,0,0x1,1,0},
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
	{TMIT_native_code, init_BB_1pYXe,(const Char *)&nmim_12,6,
	(const Char *)&sgim_12,75,1,0x1,0,xt_init_BB_1pYXe},
	{TMIT_native_code, init_BBi_Au8Qd,(const Char *)&nmim_13,6,
	(const Char *)&sgim_13,76,1,0x1,1,xt_init_BBi_Au8Qd},
	{TMIT_native_code, resolveFor_S_ap4uc,(const Char *)&nmim_14,10,
	(const Char *)&sgim_14,62,1,0x8001,2,xt_resolveFor_S_ap4uc},
    } /* end of methodsigs */
};


/*M init_BB_1pYXe: ca.mcgill.sable.soot.BuildBody.<init>(Lca/mcgill/sable/soot/BodyRepresentation;Lca/mcgill/sable/soot/BodyExpr;)V */

Class xt_init_BB_1pYXe[] = { 0 };

Void init_BB_1pYXe(Object p0, Object p1, Object p2)
{
Object a0, a1, a2;
Object av0, av1, av2;
Int i0, i1, i2;
PROLOGUE;

	av0 = p0;
	av1 = p1;
	av2 = p2;

L0:	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	init__AAyKx(a1);
	a1 = av0;
	a2 = av1;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->bodyRep = a2;
	a1 = av0;
	a2 = av2;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->sourceBuilder = a2;
	a1 = av0;
	i2 = 0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->buildBodyOptions = i2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M init_BBi_Au8Qd: ca.mcgill.sable.soot.BuildBody.<init>(Lca/mcgill/sable/soot/BodyRepresentation;Lca/mcgill/sable/soot/BodyExpr;I)V */

Class xt_init_BBi_Au8Qd[] = { 0 };

Void init_BBi_Au8Qd(Object p0, Object p1, Object p2, Int p3)
{
Object a0, a1, a2;
Object av0, av1, av2;
Int i0, i1, i2;
Int iv3;
PROLOGUE;

	av0 = p0;
	av1 = p1;
	av2 = p2;
	iv3 = p3;

L0:	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	init__AAyKx(a1);
	a1 = av0;
	a2 = av1;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->bodyRep = a2;
	a1 = av0;
	a2 = av2;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->sourceBuilder = a2;
	a1 = av0;
	i2 = iv3;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->buildBodyOptions = i2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}

/*M resolveFor_S_ap4uc: ca.mcgill.sable.soot.BuildBody.resolveFor(Lca/mcgill/sable/soot/SootMethod;)Lca/mcgill/sable/soot/Body; */

Class xt_resolveFor_S_ap4uc[] = { 0 };

Object resolveFor_S_ap4uc(Object p0, Object p1)
{
Object a0, a1, a2, a3, a4;
Object av0, av1;
Int i0, i1, i2, i3, i4;
PROLOGUE;

	av0 = p0;
	av1 = p1;

L0:	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	a1 = ((struct in_ca_mcgill_sable_soot_BuildBody*)a1)->bodyRep;
	a2 = av1;
	a3 = av0;
	if (!a3) { 
		SetNPESource(); goto NULLX;
	}
	a3 = ((struct in_ca_mcgill_sable_soot_BuildBody*)a3)->sourceBuilder;
	a4 = av1;
	if (!a3) { 
		SetNPESource(); goto NULLX;
	}
	a3 = (*(Object(*)())findinterface(a3,-2117377626)->f)(a3,a4);
	a4 = av0;
	if (!a4) { 
		SetNPESource(); goto NULLX;
	}
	i4 = ((struct in_ca_mcgill_sable_soot_BuildBody*)a4)->buildBodyOptions;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	a1 = (*(Object(*)())findinterface(a1,1529379912)->f)(a1,a2,a3,i4);
	return a1;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}



const Char ch_ca_mcgill_sable_soot_BuildBody[] = {  /* string pool */'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','B','u','i','l','d','B','o','d','y'};

const void *st_ca_mcgill_sable_soot_BuildBody[] = {
    ch_ca_mcgill_sable_soot_BuildBody+30,	/* 0. ca.mcgill.sable.soot.BuildBody */
    0};
