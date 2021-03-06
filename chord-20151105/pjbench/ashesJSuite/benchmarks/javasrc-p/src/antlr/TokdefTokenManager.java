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
import java.io.*;
import java.util.Hashtable;
import java.util.Enumeration;
import antlr.collections.impl.Vector;

/** Static implementation of the TokenManager, used for tokdef option 
 */
class TokdefTokenManager extends SimpleTokenManager {
	private int dummy;
	private String filename;
	private boolean noDefine = false;		// allow new tokens with tokdef
	protected Grammar grammar;


	TokdefTokenManager(Grammar grammar, String filename_, Tool tool_) {
		// initialize
		super("", tool_);
		this.grammar = grammar;
//		noDefine = true;
		filename = filename_;

		// Read a file with lines of the form ID=number
		try {
			// SAS: changed the following for proper text io
			FileReader fileIn = new FileReader(filename);
			ANTLRTokdefLexer tokdefLexer = new ANTLRTokdefLexer(fileIn);
			ANTLRTokdefParser tokdefParser = new ANTLRTokdefParser(tokdefLexer);
			tokdefParser.file(this);
		} 
		catch (ParserException ex) {
			tool.panic("Error parsing tokdef file '" + filename + "': " + ex.toString());
		}
		catch (IOException ex) {
			tool.panic("Error reading tokdef file '" + filename + "'");
		}
	}
	/** define a token. */
	public void define(TokenSymbol ts) {
		if (noDefine) {
			tool.error("New token type defined when using tokdef option");
		} else {
			super.define(ts);
		}
		// Allow processing to continue anyway
	}
	/** define a token.  Intended for use only when reading the tokdef file. */
	public void define(String s, int ttype) {
		TokenSymbol ts=null;
		if ( s.startsWith("\"") ) {
			ts = new StringLiteralSymbol(s);
		}
		else {
			ts = new TokenSymbol(s);
		}	
		ts.setTokenType(ttype);
		super.define(ts);
		maxToken = (ttype+1)>maxToken ? (ttype+1) : maxToken;	// record maximum token type
	}
	public String getName() { return grammar.getClassName(); }
	/** tokdef token manager is read-only if output would be same as input */
	public boolean isReadOnly() {
		return filename.equals(grammar.getClassName()+"TokenTypes.txt");
	}
	/** Get the next unused token type.  Invalid for tokdefs. */
	public int nextTokenType() {
		if ( noDefine ) {
			tool.error("New token type defined when using tokdef option");
			// Return error value
			return 0;
		}
		return super.nextTokenType();	
	}
}
