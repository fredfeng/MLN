/*  ca_mcgill_sable_soot_coffi_Instruction_Iconst_1.c -- from Java class ca.mcgill.sable.soot.coffi.Instruction_Iconst_1  */
/*  created by Toba  */

#include "toba.h"
#include "ca_mcgill_sable_soot_coffi_Instruction_Iconst_1.h"
#include "ca_mcgill_sable_soot_coffi_Instruction_noargs.h"
#include "ca_mcgill_sable_soot_coffi_Instruction.h"
#include "java_lang_Cloneable.h"
#include "java_lang_Object.h"
#include "java_lang_String.h"
#include "java_lang_Class.h"
#include "ca_mcgill_sable_soot_coffi_ByteCode.h"

static const Class supers[] = {
    &cl_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1.C,
    &cl_ca_mcgill_sable_soot_coffi_Instruction_noargs.C,
    &cl_ca_mcgill_sable_soot_coffi_Instruction.C,
    &cl_java_lang_Object.C,
};

static const Class inters[] = {
    &cl_java_lang_Cloneable.C,
};

static const Class others[] = {
    &cl_ca_mcgill_sable_soot_coffi_ByteCode.C,
};

extern const Char ch_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1[];
extern const void *st_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1[];
extern Class xt_init__sBNqD[];

#define HASHMASK 0x0
/*  0.  c2aafd4e  (0)  equals  */
static const struct ihash htable[2] = {
    -1028981426, &cl_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1.M.equals_O_Ba6e0,
    0, 0,
};

static const CARRAY(47) nmchars = {&acl_char, 0, 47, 0,
'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','c','o','f','f','i','.','I','n','s','t','r','u','c','t','i',
'o','n','_','I','c','o','n','s','t','_','1'};
static struct in_java_lang_String classname =
    { &cl_java_lang_String, 0, (Object)&nmchars, 0, 47 };
static const Char nmiv_0[] = {
'c','o','d','e'};
static const Char sgiv_0[] = {
'B'};
static const Char nmiv_1[] = {
'l','a','b','e','l'};
static const Char sgiv_1[] = {
'I'};
static const Char nmiv_2[] = {
'n','a','m','e'};
static const Char sgiv_2[] = {
'L','j','a','v','a','/','l','a','n','g','/','S','t','r','i','n','g',';'};
static const Char nmiv_3[] = {
'n','e','x','t'};
static const Char sgiv_3[] = {
'L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/','s',
'o','o','t','/','c','o','f','f','i','/','I','n','s','t','r','u','c','t',
'i','o','n',';'};
static const Char nmiv_4[] = {
'l','a','b','e','l','l','e','d'};
static const Char sgiv_4[] = {
'Z'};
static const Char nmiv_5[] = {
'b','r','a','n','c','h','e','s'};
static const Char sgiv_5[] = {
'Z'};
static const Char nmiv_6[] = {
'c','a','l','l','s'};
static const Char sgiv_6[] = {
'Z'};
static const Char nmiv_7[] = {
'r','e','t','u','r','n','s'};
static const Char sgiv_7[] = {
'Z'};
static const Char nmiv_8[] = {
'o','r','i','g','i','n','a','l','I','n','d','e','x'};
static const Char sgiv_8[] = {
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
'(','B',')','V'};
static const Char nmim_13[] = {
'p','a','r','s','e'};
static const Char sgim_13[] = {
'(','[','B','I',')','I'};
static const Char nmim_14[] = {
'c','o','m','p','i','l','e'};
static const Char sgim_14[] = {
'(','[','B','I',')','I'};
static const Char nmim_15[] = {
'o','f','f','s','e','t','T','o','P','o','i','n','t','e','r'};
static const Char sgim_15[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','c','o','f','f','i','/','B','y','t','e','C','o','d',
'e',';',')','V'};
static const Char nmim_16[] = {
'n','e','x','t','O','f','f','s','e','t'};
static const Char sgim_16[] = {
'(','I',')','I'};
static const Char nmim_17[] = {
'b','r','a','n','c','h','p','o','i','n','t','s'};
static const Char sgim_17[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','c','o','f','f','i','/','I','n','s','t','r','u','c',
't','i','o','n',';',')','[','L','c','a','/','m','c','g','i','l','l','/',
's','a','b','l','e','/','s','o','o','t','/','c','o','f','f','i','/','I',
'n','s','t','r','u','c','t','i','o','n',';'};
static const Char nmim_18[] = {
'm','a','r','k','C','P','R','e','f','s'};
static const Char sgim_18[] = {
'(','[','Z',')','V'};
static const Char nmim_19[] = {
'r','e','d','i','r','e','c','t','C','P','R','e','f','s'};
static const Char sgim_19[] = {
'(','[','S',')','V'};
static const Char nmim_20[] = {
'e','q','u','a','l','s'};
static const Char sgim_20[] = {
'(','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','c','o','f','f','i','/','I','n','s','t','r','u','c',
't','i','o','n',';',')','Z'};
static const Char nmim_21[] = {
't','o','S','t','r','i','n','g'};
static const Char sgim_21[] = {
'(','[','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e',
'/','s','o','o','t','/','c','o','f','f','i','/','c','p','_','i','n','f',
'o',';',')','L','j','a','v','a','/','l','a','n','g','/','S','t','r','i',
'n','g',';'};

static struct vt_generic cv_table[] = {
    {0}
};

#ifndef offsetof
#define offsetof(s,m) ((int)&(((s *)0))->m)
#endif
static struct vt_generic iv_table[] = {
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, code), 0,(const Char *)&nmiv_0,4,(const Char *)&sgiv_0,1,0,0x1,2}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, label), 0,(const Char *)&nmiv_1,5,(const Char *)&sgiv_1,1,0,0x1,3}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, name), 0,(const Char *)&nmiv_2,4,(const Char *)&sgiv_2,18,0,0x1,4}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, next), 0,(const Char *)&nmiv_3,4,(const Char *)&sgiv_3,40,0,0x1,5}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, labelled), 0,(const Char *)&nmiv_4,8,(const Char *)&sgiv_4,1,0,0x1,6}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, branches), 0,(const Char *)&nmiv_5,8,(const Char *)&sgiv_5,1,0,0x1,7}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, calls), 0,(const Char *)&nmiv_6,5,(const Char *)&sgiv_6,1,0,0x1,8}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, returns), 0,(const Char *)&nmiv_7,7,(const Char *)&sgiv_7,1,0,0x1,9}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1, originalIndex), 0,(const Char *)&nmiv_8,13,(const Char *)&sgiv_8,1,0,0x0,10}, 
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
} inr_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1 = {
  (struct cl_generic *)&cl_toba_classfile_ClassRef.C, 0, &classname, &cl_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1.C.classclass, 0};

struct cl_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1 cl_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1 = { {
    1, 0,
    &classname,
    &cl_java_lang_Class.C, 0,
    sizeof(struct in_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1),
    22,
    0,
    9,
    0,
    4, supers,
    1, 0, inters, HASHMASK, htable,
    1, others,
    0, 0,
    ch_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1,
    st_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1,
    0,
    init__sBNqD,
    finalize__UKxhs,
    0,
    0,
    43,
    0x20,
    0,
    (struct in_toba_classfile_ClassRef *)&inr_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1,
    iv_table, cv_table,
    sm_table},
    { /* methodsigs */
	{TMIT_native_code, init__sBNqD,(const Char *)&nmim_0,6,
	(const Char *)&sgim_0,3,1,0x1,0,xt_init__sBNqD},
	{TMIT_native_code, clone__C5Kdq,(const Char *)&nmim_1,5,
	(const Char *)&sgim_1,20,0,0x4,1,0},
	{TMIT_native_code, equals_O_Ba6e0,(const Char *)&nmim_2,6,
	(const Char *)&sgim_2,21,0,0x8001,3,0},
	{TMIT_native_code, finalize__UKxhs,(const Char *)&nmim_3,8,
	(const Char *)&sgim_3,3,0,0x4,4,0},
	{TMIT_native_code, getClass__zh19H,(const Char *)&nmim_4,8,
	(const Char *)&sgim_4,19,0,0x111,5,0},
	{TMIT_native_code, hashCode__P84mQ,(const Char *)&nmim_5,8,
	(const Char *)&sgim_5,3,0,0x1,10,0},
	{TMIT_native_code, notify__ne4bk,(const Char *)&nmim_6,6,
	(const Char *)&sgim_6,3,0,0x111,7,0},
	{TMIT_native_code, notifyAll__iTnlk,(const Char *)&nmim_7,9,
	(const Char *)&sgim_7,3,0,0x111,8,0},
	{TMIT_native_code, toString__dkN89,(const Char *)&nmim_8,8,
	(const Char *)&sgim_8,20,0,0x1,2,0},
	{TMIT_native_code, wait__Zlr2b,(const Char *)&nmim_9,4,
	(const Char *)&sgim_9,3,0,0x11,11,0},
	{TMIT_native_code, wait_l_1Iito,(const Char *)&nmim_10,4,
	(const Char *)&sgim_10,4,0,0x111,12,0},
	{TMIT_native_code, wait_li_07Ea2,(const Char *)&nmim_11,4,
	(const Char *)&sgim_11,5,0,0x11,13,0},
	{TMIT_native_code, init_b_Qlb3v,(const Char *)&nmim_12,6,
	(const Char *)&sgim_12,4,0,0x1,0,0},
	{TMIT_native_code, parse_abi_OOtf7,(const Char *)&nmim_13,5,
	(const Char *)&sgim_13,6,0,0x1,1,0},
	{TMIT_native_code, compile_abi_GLznz,(const Char *)&nmim_14,7,
	(const Char *)&sgim_14,6,0,0x1,2,0},
	{TMIT_native_code, offsetToPointer_B_Myq1j,(const Char *)&nmim_15,15,
	(const Char *)&sgim_15,40,0,0x1,5,0},
	{TMIT_native_code, nextOffset_i_BhJRD,(const Char *)&nmim_16,10,
	(const Char *)&sgim_16,4,0,0x1,6,0},
	{TMIT_native_code, branchpoints_I_dvskP,(const Char *)&nmim_17,12,
	(const Char *)&sgim_17,83,0,0x1,7,0},
	{TMIT_native_code, markCPRefs_az_4cO9Q,(const Char *)&nmim_18,10,
	(const Char *)&sgim_18,5,0,0x1,8,0},
	{TMIT_native_code, redirectCPRefs_as_4OU8w,(const Char *)&nmim_19,14,
	(const Char *)&sgim_19,5,0,0x1,9,0},
	{TMIT_native_code, equals_I_LuRCj,(const Char *)&nmim_20,6,
	(const Char *)&sgim_20,43,0,0x1,11,0},
	{TMIT_native_code, toString_ac_9JP2g,(const Char *)&nmim_21,8,
	(const Char *)&sgim_21,57,0,0x1,16,0},
    } /* end of methodsigs */
};


/*M init__sBNqD: ca.mcgill.sable.soot.coffi.Instruction_Iconst_1.<init>()V */

Class xt_init__sBNqD[] = { 0 };

Void init__sBNqD(Object p0)
{
Object a0, a1, a2;
Object av0;
Int i0, i1, i2;
PROLOGUE;

	av0 = p0;

L0:	a1 = av0;
	i2 = 4;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	init_b_Qlb3v(a1,i2);
	a1 = av0;
	a2 = (Object)st_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1[1];
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	((struct in_ca_mcgill_sable_soot_coffi_Instruction*)a1)->name = a2;
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}



const Char ch_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1[] = {  /* string pool */'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','c','o','f','f','i','.','I','n','s','t','r','u','c','t','i',
'o','n','_','I','c','o','n','s','t','_','1','i','c','o','n','s','t','_',
'1'};

const void *st_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1[] = {
    ch_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1+47,	/* 0. ca.mcgill.sable.soot.coffi.Instruction_I */
    ch_ca_mcgill_sable_soot_coffi_Instruction_Iconst_1+55,	/* 1. iconst_1 */
    0};
