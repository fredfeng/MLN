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
/**A Stream of characters fed to the lexer from a InputStream that can
 * be rewound via mark()/rewind() methods.
 * <p>
 * A dynamic array is used to buffer up all the input characters.  Normally,
 * "k" characters are stored in the buffer.  More characters may be stored during
 * guess mode (testing syntactic predicate), or when LT(i>k) is referenced.
 * Consumption of characters is deferred.  In other words, reading the next
 * character is not done by conume(), but deferred until needed by LA or LT.
 * <p>
 *
 * @see antlr.CharQueue
 */

// SAS: Added this class to genericise the input buffers for scanners
//      This allows a scanner to use a binary (FileInputStream) or
//      text (FileReader) stream of data; the generated scanner
//      subclass will define the input stream
//      There are two subclasses to this: CharBuffer and ByteBuffer
import java.io.IOException;

public abstract class InputBuffer {
	// char source
//	transient Reader input;  // leave to subclasses

	// Number of active markers
	int nMarkers = 0;

	// Additional offset used when markers are active
	int markerOffset = 0;

	// Number of calls to consume() since last LA() or LT() call
	int numToConsume = 0;

	// Circular queue
	CharQueue queue;

	/** Create an input buffer */
	public InputBuffer() {
		queue = new CharQueue(1);
	}
	/** This method updates the state of the input buffer so that
	 *  the text matched since the most recent mark() is no longer
	 *  held by the buffer.  So, you either do a mark/rewind for
	 *  failed predicate or mark/commit to keep on parsing without
	 *  rewinding the input.
	 */
	public void commit() {
		nMarkers--;
	}
	/** Mark another character for deferred consumption */
	public void consume() {
		numToConsume++;
	}
	/** Ensure that the input buffer is sufficiently full */
	public abstract void fill(int amount) throws IOException;
	public String getLAChars() {
		StringBuffer la = new StringBuffer();
		for(int i = markerOffset; i < queue.nbrEntries; i++)
			la.append(queue.elementAt(i));
		return la.toString();
	}
	public String getMarkedChars() {
		StringBuffer marked = new StringBuffer();
		for(int i = 0; i < markerOffset; i++)
			marked.append(queue.elementAt(i));
		return marked.toString();
	}
	public boolean isMarked() {
		return (nMarkers != 0);
	}
	/** Get a lookahead character */
	public char LA(int i) throws IOException {
		fill(i);
		return queue.elementAt(markerOffset + i - 1);
	}
	/**Return an integer marker that can be used to rewind the buffer to
	 * its current state.
	 */
	public int mark() {
		syncConsume();
		nMarkers++;
		return markerOffset;
	}
	/**Rewind the character buffer to a marker.
	 * @param mark Marker returned previously from mark()
	 */
	public void rewind(int mark) {
		syncConsume();
		markerOffset = mark;
		nMarkers--;
	}
	/** Sync up deferred consumption */
	protected void syncConsume() {
		while (numToConsume > 0) {
			if (nMarkers > 0)
			{
				// guess mode -- leave leading characters and bump offset.
				markerOffset++;
			} else {
				// normal mode -- remove first character
				queue.removeFirst();
			}
			numToConsume--;
		}
	}
}
