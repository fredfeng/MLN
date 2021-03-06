header {
package antlr.actions;
}

{
import java.io.StringReader;
import antlr.collections.impl.Vector;
import antlr.*;
}

/**
 * SOFTWARE RIGHTS
 * 
 * ANTLR 2.5.0 MageLang Insitute, 1998
 * 
 * We reserve no legal rights to the ANTLR--it is fully in the
 * public domain. An individual or company may do whatever
 * they wish with source code distributed with ANTLR or the
 * code generated by ANTLR, including the incorporation of
 * ANTLR, or its output, into commerical software.
 * 
 * We encourage users to develop software with ANTLR. However,
 * we do ask that credit is given to us for developing
 * ANTLR. By "credit", we mean that if you use ANTLR or
 * incorporate any source code into one of your programs
 * (commercial product, research project, or otherwise) that
 * you acknowledge this fact somewhere in the documentation,
 * research report, etc... If you like ANTLR and have
 * developed a nice tool with the output, please mention that
 * you developed it using ANTLR. In addition, we ask that the
 * headers remain intact in our source code. As long as these
 * guidelines are kept, we expect to continue enhancing this
 * system and expect to make other tools available as they are
 * completed.
 * 
 * The ANTLR gang:
 * @version ANTLR 2.5.0 MageLang Insitute, 1998
 * @author Terence Parr, MageLang Institute; http://www.MageLang.com
 * @author John Lilley, Empathy Software; http://www.Empathy.com
 *
 * Perform the following translations:

    AST related translations

	##				-> currentRule_AST
	#(x,y,z)		-> codeGenerator.getASTCreateString(vector-of(x,y,z))
	#[x]			-> codeGenerator.getASTCreateString(x)
	#x				-> codeGenerator.mapTreeId(x)

	Inside context of #(...), you can ref (x,y,z), [x], and x as shortcuts.

    Text related translations

	$append(x)		-> text.append(x)
	$setText(x)		-> text.setLength(_begin); text.append(x)
	$getText		-> new String(text.getBuffer(),_begin,text.length()-_begin)
	$setToken(x)	-> _token = x
	$setType(x)		-> _ttype = x
 */
class ActionLexer extends Lexer;
options {
	k=2;
	charVocabulary='\3'..'\176';
	testLiterals=false;
	interactive=true;
}

{
	protected RuleBlock currentRule;
	protected CodeGenerator generator;
	protected int lineOffset = 0;
	private Tool tool;	// The ANTLR tool
	ActionTransInfo transInfo;

 	public ActionLexer( String s,
						RuleBlock currentRule,
						CodeGenerator generator,
						ActionTransInfo transInfo) {
		this(new StringReader(s));
		this.currentRule = currentRule;
		this.generator = generator;
		this.transInfo = transInfo;
	}

	public void setLineOffset(int lineOffset) {
		// this.lineOffset = lineOffset;
		setLine(lineOffset);
	}

	public void setTool(Tool tool) {
		this.tool = tool;
	}

	// Override of error-reporting for syntax
	public void reportError(ScannerException e) {
		System.err.print("Syntax error in action: ");
		super.reportError(e);
	}

}

// rules are protected because we don't care about nextToken().

public
ACTION
	:	(	STUFF
		|	AST_ITEM
		)+
	;

// stuff in between #(...) and #id items
protected
STUFF
	:	COMMENT
	|	STRING
	|	CHAR
	|	'\r' ('\n')?	{newline();}
	|	'\n'		{newline();}
	|	'/'	~('/'|'*')	// non-comment start '/'
	|	( ~('/'|'\n'|'\r'|'$'|'#'|'"'|'\'') )+
	;

protected
AST_ITEM
	:	'#'! t:TREE
	|	'#'! id:ID
		{
		String idt = id.getText();
		$setText(generator.mapTreeId(idt,transInfo));
		}
		(WS)?
		( VAR_ASSIGN )?
	|	'#'! ctor:AST_CONSTRUCTOR
	|	"##"
		{
		String r=currentRule.getRuleName()+"_AST"; $setText(r);
		if ( transInfo!=null ) {
			transInfo.refRuleRoot=r;	// we ref root of tree
		}
		}
		(WS)?
		( VAR_ASSIGN )?
	|	TEXT_ITEM
	;

protected
TEXT_ITEM
	:	"$append(" a1:TEXT_ARG ')'
		{String t = "text.append("+a1.getText()+")"; $setText(t);}
	|	"$set"
		(	"Text(" a2:TEXT_ARG ')'
			{
			String t;
			if (generator instanceof CppCodeGenerator) {
				t="text.erase(_begin); text.append("+a2.getText()+")";
			} else {
				t="text.setLength(_begin); text.append("+a2.getText()+")";
			}
			$setText(t);
			}
		|	"Token(" a3:TEXT_ARG ')'
			{
			String t="_token = "+a3.getText();
			$setText(t);
			}
		|	"Type(" a4:TEXT_ARG ')'
			{
			String t="_ttype = "+a4.getText();
			$setText(t);
			}
		)
	|	"$getText"
		{
			if (generator instanceof CppCodeGenerator) {
				$setText("text.substr(_begin,text.length()-_begin)");
			} else {
				$setText("new String(text.getBuffer(),_begin,text.length()-_begin)");
			}
		}
	;

protected
TREE!
{
	StringBuffer buf = new StringBuffer();
	int n=0;
	Vector terms = new Vector(10);
}
	:	'('
		(WS)?
		t:TREE_ELEMENT {terms.appendElement(t.getText());}
		(WS)?
		(	','	(WS)?
			t2:TREE_ELEMENT {terms.appendElement(t2.getText());}
			(WS)?
		)*
		{$setText(generator.getASTCreateString(terms));}
		')'
	;

protected
TREE_ELEMENT
	:	'#'! TREE
	|	'#'! AST_CONSTRUCTOR
	|	'#'! id:ID_ELEMENT
		{String t=generator.mapTreeId(id.getText(), null); $setText(t);}
	|	"##"
		{String t = currentRule.getRuleName()+"_AST"; $setText(t);}
	|	TREE
	|	AST_CONSTRUCTOR
	|	ID_ELEMENT
	;

protected
AST_CONSTRUCTOR!
	:	'[' (WS)? x:AST_CTOR_ELEMENT (WS)?
		(',' (WS)? y:AST_CTOR_ELEMENT (WS)? )? ']'
		{
		String ys = "";
		if ( y!=null ) {
			ys = ","+y.getText();
		}
		$setText(generator.getASTCreateString(x.getText()+ys));
		}
	;

/** The arguments of a #[...] constructor are text, token type,
 *  or a tree.
 */
protected
AST_CTOR_ELEMENT
	:	STRING
	|	INT
	|	TREE_ELEMENT
	;

/** An ID_ELEMENT can be a func call, array ref, simple var,
 *  or AST label ref.
 */
protected
ID_ELEMENT
	:	id:ID (WS!)?
		(	'(' (WS!)? ( ARG (',' (WS!)? ARG)* )? (WS!)? ')'	// method call
		|	( '[' (WS!)? ARG (WS!)? ']' )+				// array reference
		|	'.' ID_ELEMENT
		|	/* could be a token reference or just a user var */
			{
			String t=generator.mapTreeId(id.getText(), transInfo);
			$setText(t);
			}
			(WS)?
			// if #rule referenced, check for assignment
			(	{transInfo!=null && transInfo.refRuleRoot!=null}?
				VAR_ASSIGN
			)?
		)
	;

protected
TEXT_ARG
	:	(TEXT_ARG_ELEMENT (WS)? )+
	;

protected
TEXT_ARG_ELEMENT
	:	TEXT_ARG_ID_ELEMENT
	|	STRING
	|	CHAR
	|	INT_OR_FLOAT
	|	TEXT_ITEM
	|	'+'
	;

protected
TEXT_ARG_ID_ELEMENT
	:	id:ID (WS!)?
		(	'(' (WS!)? ( TEXT_ARG (',' TEXT_ARG)* )* (WS!)? ')'	// method call
		|	( '[' (WS!)? TEXT_ARG (WS!)? ']' )+				// array reference
		|	'.' TEXT_ARG_ID_ELEMENT
		|	"->" TEXT_ARG_ID_ELEMENT
		|	"::" TEXT_ARG_ID_ELEMENT
		|
		)
	;

protected
ARG	:	(	TREE_ELEMENT
		|	STRING
		|	CHAR
		|	INT_OR_FLOAT
		)
		( (WS)? ( '+'| '-' | '*' | '/' ) (WS)? ARG )*
	;

protected
ID	:	('a'..'z'|'A'..'Z'|'_')(('a'..'z'|'A'..'Z'|'0'..'9'|'_'))*
	;

protected
VAR_ASSIGN
	:	(	'=' c:~'='
			{
			// inform the code generator that an assignment was done to
			// AST root for the rule if invoker set refRuleRoot.
			if ( transInfo!=null && transInfo.refRuleRoot!=null ) {
				transInfo.assignToRoot=true;
			}
			}
		|	"=="
		)
	;

protected
COMMENT
	:	SL_COMMENT
	|	ML_COMMENT
	;

protected
SL_COMMENT : 
	"//" 
	(~('\n'|'\r'))* ('\n'|'\r'('\n')?)
	{ newline(); }
	;

protected
ML_COMMENT :
	"/*"
	(
		{ LA(2)!='/' }? '*'
	|	'\r' ('\n')?	{newline();}
	|	'\n'		{newline();}
	|	~('*'|'\n'|'\r')
	)*
	"*/"
	;

protected
CHAR :	
	'\'' 
	( ESC | ~'\'' ) 
	'\''
	;

protected
STRING :	
	'"' 
	(ESC|~'"')* 
	'"'
	;

protected
ESC	:	'\\'
		(	'n'
		|	'r'
		|	't'
		|	'b'
		|	'f'
		|	'"'
		|	'\''
		|	'\\'
		|	('0'..'3') ( DIGIT (DIGIT)? )?
		|	('4'..'7') (DIGIT)?
		)
	;

protected
DIGIT
	:	'0'..'9'
	;

protected
INT	:	(DIGIT)+
	;

protected
INT_OR_FLOAT
	:	(DIGIT)+
		(	'.' (DIGIT)*
		|	'L'
		|	'l'
		)?
	;

protected
WS	:	(	' '
		|	'\t'
		|	'\r' ('\n')?	{newline();}
		|	'\n'		{newline();}
		)+
	;
