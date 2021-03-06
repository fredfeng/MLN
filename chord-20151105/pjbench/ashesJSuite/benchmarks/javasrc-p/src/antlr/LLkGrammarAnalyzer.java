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

public interface LLkGrammarAnalyzer extends GrammarAnalyzer {


	public boolean deterministic(AlternativeBlock blk);
	public boolean deterministic(OneOrMoreBlock blk);
	public boolean deterministic(ZeroOrMoreBlock blk);
	public Lookahead FOLLOW(int k, RuleEndElement end);
	public Lookahead look(int k, ActionElement action);
	public Lookahead look(int k, AlternativeBlock blk);
	public Lookahead look(int k, BlockEndElement end);
	public Lookahead look(int k, CharLiteralElement atom);
	public Lookahead look(int k, CharRangeElement end);
	public Lookahead look(int k, GrammarAtom atom);
	public Lookahead look(int k, OneOrMoreBlock blk);
	public Lookahead look(int k, RuleBlock blk);
	public Lookahead look(int k, RuleEndElement end);
	public Lookahead look(int k, RuleRefElement rr);
	public Lookahead look(int k, StringLiteralElement atom);
	public Lookahead look(int k, SynPredBlock blk);
	public Lookahead look(int k, TokenRangeElement end);
	public Lookahead look(int k, TreeElement end);
	public Lookahead look(int k, WildcardElement wc);
	public Lookahead look(int k, ZeroOrMoreBlock blk);
	public Lookahead look(int k, String rule);
	public void setGrammar(Grammar g);
	public boolean subruleCanBeInverted(AlternativeBlock blk, boolean forLexer);
}
