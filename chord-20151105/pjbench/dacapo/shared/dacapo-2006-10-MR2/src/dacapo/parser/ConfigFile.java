/* Generated By:JavaCC: Do not edit this line. ConfigFile.java */
package dacapo.parser;

import java.io.*;
import java.util.Vector;

class ConfigFile implements ConfigFileConstants {
  public static void main(String[] args) {
    ConfigFile parser;
    try {
      parser = new ConfigFile(new FileInputStream(args[0]));
      parser.configFile();
    } catch (ParseException p) {
        System.err.println("Parse exception");
        p.printStackTrace();
        return;
    } catch (FileNotFoundException e) {
        System.err.println("File " + args[0] + " not found.");
        return;
    }
    System.out.println("Success!");
  }

  private static String unQuote(Token token) {
        return token.image.substring(1,token.image.length()-1);
  }

  private static String unHex(Token token) {
        return token.image.substring(2).toLowerCase();
  }

  final public Config configFile() throws ParseException {
  Config config;
    config = config();
    label_1:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case SIZE:
        sizeSpec(config);
        break;
      case DESCRIPTION:
        description(config);
        break;
      default:
        jj_la1[0] = jj_gen;
        jj_consume_token(-1);
        throw new ParseException();
      }
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case DESCRIPTION:
      case SIZE:
        ;
        break;
      default:
        jj_la1[1] = jj_gen;
        break label_1;
      }
    }
    {if (true) return config;}
    throw new Error("Missing return statement in function");
  }

  final public Config config() throws ParseException {
  Config config; Token name, className, methodName;
    jj_consume_token(BENCHMARK);
    name = jj_consume_token(IDENT);
    config = new Config(name.image);
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case CLASS:
      jj_consume_token(CLASS);
      className = jj_consume_token(IDENT);
      config.setClass(className.image);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case METHOD:
        jj_consume_token(METHOD);
        methodName = jj_consume_token(IDENT);
        config.setMethod(methodName.image);
        break;
      default:
        jj_la1[2] = jj_gen;
        ;
      }
      break;
    default:
      jj_la1[3] = jj_gen;
      ;
    }
    jj_consume_token(SEMI);
    {if (true) return config;}
    throw new Error("Missing return statement in function");
  }

  final public void sizeSpec(Config config) throws ParseException {
  Token size;
    jj_consume_token(SIZE);
    size = jj_consume_token(IDENT);
    label_2:
    while (true) {
      sizeClause(config,size.image);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case ARGS:
      case OUTPUT:
        ;
        break;
      default:
        jj_la1[4] = jj_gen;
        break label_2;
      }
    }
    jj_consume_token(SEMI);
  }

  final public void sizeClause(Config config, String size) throws ParseException {
    Vector args = new Vector();
        String arg;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case ARGS:
      jj_consume_token(ARGS);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case STRING_LITERAL:
        arg = string();
                                  args.add(arg);
        label_3:
        while (true) {
          switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
          case COMMA:
            ;
            break;
          default:
            jj_la1[5] = jj_gen;
            break label_3;
          }
          jj_consume_token(COMMA);
          arg = string();
                                          args.add(arg);
        }
        break;
      default:
        jj_la1[6] = jj_gen;
        ;
      }
         config.addSize(size,args);
      break;
    case OUTPUT:
      jj_consume_token(OUTPUT);
      outputFile(config,size);
      label_4:
      while (true) {
        switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
        case COMMA:
          ;
          break;
        default:
          jj_la1[7] = jj_gen;
          break label_4;
        }
        jj_consume_token(COMMA);
        outputFile(config,size);
      }
      break;
    default:
      jj_la1[8] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
  }

  final public void outputFile(Config config, String size) throws ParseException {
  String file;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case STRING_LITERAL:
      file = string();
      break;
    case STDOUT:
      jj_consume_token(STDOUT);
                     file = "stdout.log";
      break;
    case STDERR:
      jj_consume_token(STDERR);
                     file = "stderr.log";
      break;
    default:
      jj_la1[9] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
          config.addOutputFile(size,file);
    label_5:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case BYTES:
      case DIGEST:
      case EXISTS:
      case KEEP:
      case LINES:
        ;
        break;
      default:
        jj_la1[10] = jj_gen;
        break label_5;
      }
      outputClause(config,size,file);
    }
  }

  final public void outputClause(Config config, String size, String file) throws ParseException {
  String digest; int n; long l;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case DIGEST:
      jj_consume_token(DIGEST);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case LPAREN:
        jj_consume_token(LPAREN);
        digestOption(config,size,file);
        label_6:
        while (true) {
          switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
          case COMMA:
            ;
            break;
          default:
            jj_la1[11] = jj_gen;
            break label_6;
          }
          jj_consume_token(COMMA);
          digestOption(config,size,file);
        }
        jj_consume_token(RPAREN);
        break;
      default:
        jj_la1[12] = jj_gen;
        ;
      }
      digest = hex();
                         config.setDigest(size,file,digest);
      break;
    case EXISTS:
      jj_consume_token(EXISTS);
                                  config.setExists(size,file);
      break;
    case KEEP:
      jj_consume_token(KEEP);
                                  config.setKeep(size,file);
      break;
    case LINES:
      jj_consume_token(LINES);
      n = integer();
                                  config.setLines(size,file,n);
      break;
    case BYTES:
      jj_consume_token(BYTES);
      l = longInt();
                                  config.setBytes(size,file,l);
      break;
    default:
      jj_la1[13] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
  }

  final public void digestOption(Config config, String size, String file) throws ParseException {
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case TEXT:
      jj_consume_token(TEXT);
                          config.setTextFile(size,file,true);
      break;
    case BINARY:
      jj_consume_token(BINARY);
                          config.setTextFile(size,file,false);
      break;
    case FILTER:
      jj_consume_token(FILTER);
                          config.setFilterScratch(size,file,true);
      break;
    case RAW:
      jj_consume_token(RAW);
                                  config.setFilterScratch(size,file,false);
      break;
    default:
      jj_la1[14] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
  }

  final public void description(Config config) throws ParseException {
    jj_consume_token(DESCRIPTION);
    descElement(config);
    label_7:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case COMMA:
        ;
        break;
      default:
        jj_la1[15] = jj_gen;
        break label_7;
      }
      jj_consume_token(COMMA);
      descElement(config);
    }
    jj_consume_token(SEMI);
  }

  final public void descElement(Config config) throws ParseException {
  String id, desc;
    id = descId();
    desc = string();
    config.addDesc(id,desc);
  }

  final public String descId() throws ParseException {
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case AUTHOR:
      jj_consume_token(AUTHOR);
      break;
    case COPYRIGHT:
      jj_consume_token(COPYRIGHT);
      break;
    case LICENSE:
      jj_consume_token(LICENSE);
      break;
    case LONG:
      jj_consume_token(LONG);
      break;
    case SHORT:
      jj_consume_token(SHORT);
      break;
    case THREADS:
      jj_consume_token(THREADS);
      break;
    case REPEATS:
      jj_consume_token(REPEATS);
      break;
    case URL:
      jj_consume_token(URL);
      break;
    case VERSION:
      jj_consume_token(VERSION);
      break;
    default:
      jj_la1[16] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
    {if (true) return token.image;}
    throw new Error("Missing return statement in function");
  }

  final public String string() throws ParseException {
  Token s;
    s = jj_consume_token(STRING_LITERAL);
                             {if (true) return unQuote(s);}
    throw new Error("Missing return statement in function");
  }

  final public int integer() throws ParseException {
  Token n;
    n = jj_consume_token(INT_LITERAL);
                          {if (true) return Integer.parseInt(n.image);}
    throw new Error("Missing return statement in function");
  }

  final public long longInt() throws ParseException {
  Token n;
    n = jj_consume_token(INT_LITERAL);
                          {if (true) return Long.parseLong(n.image);}
    throw new Error("Missing return statement in function");
  }

  final public String hex() throws ParseException {
  Token h;
    h = jj_consume_token(HEX_STRING);
                         {if (true) return unHex(h);}
    throw new Error("Missing return statement in function");
  }

  public ConfigFileTokenManager token_source;
  SimpleCharStream jj_input_stream;
  public Token token, jj_nt;
  private int jj_ntk;
  private int jj_gen;
  final private int[] jj_la1 = new int[17];
  static private int[] jj_la1_0;
  static private int[] jj_la1_1;
  static {
      jj_la1_0();
      jj_la1_1();
   }
   private static void jj_la1_0() {
      jj_la1_0 = new int[] {0x100000,0x100000,0x10000000,0x40000,0x20002000,0x0,0x0,0x0,0x20002000,0x0,0x5620000,0x0,0x0,0x5620000,0x40810000,0x0,0x8a084000,};
   }
   private static void jj_la1_1() {
      jj_la1_1 = new int[] {0x1,0x1,0x0,0x0,0x0,0x1000,0x400,0x1000,0x0,0x406,0x0,0x1000,0x2000,0x0,0x8,0x1000,0xf0,};
   }

  public ConfigFile(java.io.InputStream stream) {
    jj_input_stream = new SimpleCharStream(stream, 1, 1);
    token_source = new ConfigFileTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  public void ReInit(java.io.InputStream stream) {
    jj_input_stream.ReInit(stream, 1, 1);
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  public ConfigFile(java.io.Reader stream) {
    jj_input_stream = new SimpleCharStream(stream, 1, 1);
    token_source = new ConfigFileTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  public void ReInit(java.io.Reader stream) {
    jj_input_stream.ReInit(stream, 1, 1);
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  public ConfigFile(ConfigFileTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  public void ReInit(ConfigFileTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 17; i++) jj_la1[i] = -1;
  }

  final private Token jj_consume_token(int kind) throws ParseException {
    Token oldToken;
    if ((oldToken = token).next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    if (token.kind == kind) {
      jj_gen++;
      return token;
    }
    token = oldToken;
    jj_kind = kind;
    throw generateParseException();
  }

  final public Token getNextToken() {
    if (token.next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    jj_gen++;
    return token;
  }

  final public Token getToken(int index) {
    Token t = token;
    for (int i = 0; i < index; i++) {
      if (t.next != null) t = t.next;
      else t = t.next = token_source.getNextToken();
    }
    return t;
  }

  final private int jj_ntk() {
    if ((jj_nt=token.next) == null)
      return (jj_ntk = (token.next=token_source.getNextToken()).kind);
    else
      return (jj_ntk = jj_nt.kind);
  }

  private java.util.Vector jj_expentries = new java.util.Vector();
  private int[] jj_expentry;
  private int jj_kind = -1;

  public ParseException generateParseException() {
    jj_expentries.removeAllElements();
    boolean[] la1tokens = new boolean[48];
    for (int i = 0; i < 48; i++) {
      la1tokens[i] = false;
    }
    if (jj_kind >= 0) {
      la1tokens[jj_kind] = true;
      jj_kind = -1;
    }
    for (int i = 0; i < 17; i++) {
      if (jj_la1[i] == jj_gen) {
        for (int j = 0; j < 32; j++) {
          if ((jj_la1_0[i] & (1<<j)) != 0) {
            la1tokens[j] = true;
          }
          if ((jj_la1_1[i] & (1<<j)) != 0) {
            la1tokens[32+j] = true;
          }
        }
      }
    }
    for (int i = 0; i < 48; i++) {
      if (la1tokens[i]) {
        jj_expentry = new int[1];
        jj_expentry[0] = i;
        jj_expentries.addElement(jj_expentry);
      }
    }
    int[][] exptokseq = new int[jj_expentries.size()][];
    for (int i = 0; i < jj_expentries.size(); i++) {
      exptokseq[i] = (int[])jj_expentries.elementAt(i);
    }
    return new ParseException(token, exptokseq, tokenImage);
  }

  final public void enable_tracing() {
  }

  final public void disable_tracing() {
  }

}
