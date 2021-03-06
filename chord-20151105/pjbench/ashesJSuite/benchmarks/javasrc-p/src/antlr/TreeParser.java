package antlr;

/**
 * <b>SOFTWARE RIGHTS</b>
 * <p>
 * ANTLR 2.5.0 MageLang Institute, 1998
 * <p>
 * We reserve no legal rights to the ANTLR--it is fully in the
 * public domain. An individual or company may do whatever
 * they wish with source code distributed with ANTLR or the
 * code generated by ANTLR, including the incorporation of
 * ANTLR, or its output, into commerical software.
 * <p>
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
 * <p>
 * The ANTLR gang:
 * @version ANTLR 2.5.0 MageLang Institute, 1998
 * @author Terence Parr, <a href=http://www.MageLang.com>MageLang Institute</a>
 * @author <br>John Lilley, <a href=http://www.Empathy.com>Empathy Software</a>
 */
import java.util.NoSuchElementException;
import antlr.collections.AST;
import antlr.collections.impl.BitSet;

public class TreeParser {
	/** The AST Null object; the parsing cursor is set to this when
	 *  it is found to be null.  This way, we can test the
	 *  token type of a node without having to have tests for null
	 *  everywhere.
	 */
	public static ASTNULLType ASTNULL = new ASTNULLType();

	/** Where did this rule leave off parsing; avoids a return parameter */
	protected AST _retTree;
	
	/** guessing nesting level; guessing==0 implies not guessing */
	protected int guessing = 0;
	
	/** Nesting level of registered handlers */
	protected int exceptionLevel = 0;
	
	/** Table of token type to token names */
	protected String[] tokenNames;

	/** AST return value for a rule is squirreled away here */
	protected AST returnAST;

	/** AST support code; parser and treeparser delegate to this object */
	protected ASTFactory astFactory = new ASTFactory();

	/** Get the AST return value squirreled away in the parser */
	public AST getAST() {
		return returnAST;
	}
	public ASTFactory getASTFactory() {
		return astFactory;
	}
	public String getTokenName(int num) {
		return tokenNames[num];
	}
	public String[] getTokenNames() {
		return tokenNames;
	}
	protected void match(AST t, int ttype) throws MismatchedTokenException {
		//System.out.println("match("+ttype+"); cursor is "+t);
		if ( t==null || t==ASTNULL || t.getType() != ttype ) {
			throw new MismatchedTokenException(getTokenNames(), t, ttype, false);
		}
	}
	/**Make sure current lookahead symbol matches the given set
	 * Throw an exception upon mismatch, which is catch by either the
	 * error handler or by the syntactic predicate.
	 */
	public void match(AST t, BitSet b) throws MismatchedTokenException {
		if ( t==null || t==ASTNULL || !b.member(t.getType()) ) {
			throw new MismatchedTokenException(getTokenNames(), t, b, false);
		}
	}
	protected void matchNot(AST t, int ttype) throws MismatchedTokenException {
		//System.out.println("match("+ttype+"); cursor is "+t);
		if ( t==null || t==ASTNULL || t.getType() == ttype ) {
			throw new MismatchedTokenException(getTokenNames(), t, ttype, true);
		}
	}
	public static void panic() {
		System.err.println("TreeWalker: panic");
		System.exit(1);
	}
	/** Parser error-reporting function can be overridden in subclass */
	public void reportError(ParserException ex) {
		System.err.println("Error: " + ex.toString());
	}
	/** Parser error-reporting function can be overridden in subclass */
	public void reportError(String s) {
		System.err.println("Error: " + s);
	}
	/** Parser warning-reporting function can be overridden in subclass */
	public void reportWarning(String s) {
		System.err.println("Warning: " + s);
	}
	/** Specify an object with support code (shared by
	 *  Parser and TreeParser.  Normally, the programmer
	 *  does not play with this, using setASTNodeType instead.
	 */
	public void setASTFactory(ASTFactory f) {
		astFactory = f;
	}
/** Specify the type of node to create during tree building */
public void setASTNodeType (String nodeType) {
	astFactory.setASTNodeType(nodeType);
}
	public void traceIn(String rname, AST t) {
		System.out.println("enter "+rname+
			"("+(t!=null?t.toString():"null")+")"+
			((guessing>0)?" [guessing]":""));
	}
	public void traceOut(String rname, AST t) {
		System.out.println("exit "+rname+
			"("+(t!=null?t.toString():"null")+")"+
			((guessing>0)?" [guessing]":""));
	}
}
