/*  ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.c -- from Java class ca.mcgill.sable.soot.coffi.LineNumberTable_attribute  */
/*  created by Toba  */

#include "toba.h"
#include "ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.h"
#include "ca_mcgill_sable_soot_coffi_attribute_info.h"
#include "java_lang_Object.h"
#include "java_lang_String.h"
#include "java_lang_Class.h"

static const Class supers[] = {
    &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.C,
    &cl_ca_mcgill_sable_soot_coffi_attribute_info.C,
    &cl_java_lang_Object.C,
};

static const Class inters[] = {
    0
};

static const Class others[] = {
    0
};

extern const Char ch_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute[];
extern const void *st_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute[];
extern Class xt_init__zOg1a[];

#define HASHMASK 0x7
/*  1.  8942e761  (1)  hashCode  */
/*  2.  4c0d6fd2  (2)  clone  */
/*  6.  c2aafd4e  (6)  equals  */
/*  7.  489f8e6f  (7)  toString  */
static const struct ihash htable[9] = {
    0, 0,
    -1992104095, &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.M.hashCode__8wJNW,
    1275949010, &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.M.clone__dslwm,
    0, 0, 0, 0, 0, 0,
    -1028981426, &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.M.equals_O_Ba6e0,
    1218416239, &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.M.toString__4d9OF,
    0, 0,
};

static const CARRAY(52) nmchars = {&acl_char, 0, 52, 0,
'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','c','o','f','f','i','.','L','i','n','e','N','u','m','b','e',
'r','T','a','b','l','e','_','a','t','t','r','i','b','u','t','e'};
static struct in_java_lang_String classname =
    { &cl_java_lang_String, 0, (Object)&nmchars, 0, 52 };
static const Char nmiv_0[] = {
'a','t','t','r','i','b','u','t','e','_','n','a','m','e'};
static const Char sgiv_0[] = {
'I'};
static const Char nmiv_1[] = {
'a','t','t','r','i','b','u','t','e','_','l','e','n','g','t','h'};
static const Char sgiv_1[] = {
'J'};
static const Char nmiv_2[] = {
'l','i','n','e','_','n','u','m','b','e','r','_','t','a','b','l','e','_',
'l','e','n','g','t','h'};
static const Char sgiv_2[] = {
'I'};
static const Char nmiv_3[] = {
'l','i','n','e','_','n','u','m','b','e','r','_','t','a','b','l','e'};
static const Char sgiv_3[] = {
'[','L','c','a','/','m','c','g','i','l','l','/','s','a','b','l','e','/',
's','o','o','t','/','c','o','f','f','i','/','l','i','n','e','_','n','u',
'm','b','e','r','_','t','a','b','l','e','_','e','n','t','r','y',';'};
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

static struct vt_generic cv_table[] = {
    {0}
};

#ifndef offsetof
#define offsetof(s,m) ((int)&(((s *)0))->m)
#endif
static struct vt_generic iv_table[] = {
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute, attribute_name), 0,(const Char *)&nmiv_0,14,(const Char *)&sgiv_0,1,0,0x1,6}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute, attribute_length), 0,(const Char *)&nmiv_1,16,(const Char *)&sgiv_1,1,0,0x1,7}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute, line_number_table_length), 0,(const Char *)&nmiv_2,24,(const Char *)&sgiv_2,1,1,0x1,0}, 
    { offsetof(struct in_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute, line_number_table), 0,(const Char *)&nmiv_3,17,(const Char *)&sgiv_3,53,1,0x1,1}, 
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
} inr_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute = {
  (struct cl_generic *)&cl_toba_classfile_ClassRef.C, 0, &classname, &cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute.C.classclass, 0};

struct cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute cl_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute = { {
    1, 0,
    &classname,
    &cl_java_lang_Class.C, 0,
    sizeof(struct in_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute),
    12,
    0,
    4,
    0,
    3, supers,
    0, 0, inters, HASHMASK, htable,
    0, others,
    0, 0,
    ch_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute,
    st_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute,
    0,
    init__zOg1a,
    finalize__UKxhs,
    0,
    0,
    43,
    0x20,
    0,
    (struct in_toba_classfile_ClassRef *)&inr_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute,
    iv_table, cv_table,
    sm_table},
    { /* methodsigs */
	{TMIT_native_code, init__zOg1a,(const Char *)&nmim_0,6,
	(const Char *)&sgim_0,3,1,0x0,0,xt_init__zOg1a},
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
    } /* end of methodsigs */
};


/*M init__zOg1a: ca.mcgill.sable.soot.coffi.LineNumberTable_attribute.<init>()V */

Class xt_init__zOg1a[] = { 0 };

Void init__zOg1a(Object p0)
{
Object a0, a1;
Object av0;
PROLOGUE;

	av0 = p0;

L0:	a1 = av0;
	if (!a1) { 
		SetNPESource(); goto NULLX;
	}
	init__ua7Tq(a1);
	return;

NULLX:
	throwNullPointerException(0);
	/*NOTREACHED*/
}



const Char ch_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute[] = {  /* string pool */'c','a','.','m','c','g','i','l','l','.','s','a','b','l','e','.','s','o',
'o','t','.','c','o','f','f','i','.','L','i','n','e','N','u','m','b','e',
'r','T','a','b','l','e','_','a','t','t','r','i','b','u','t','e'};

const void *st_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute[] = {
    ch_ca_mcgill_sable_soot_coffi_LineNumberTable_attribute+52,	/* 0. ca.mcgill.sable.soot.coffi.LineNumberTab */
    0};