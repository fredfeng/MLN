package chord.analyses.provenance.kcfa;

import gnu.trove.list.array.TIntArrayList;

import java.io.File;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import joeq.Class.jq_ClassInitializer;
import joeq.Class.jq_Field;
import joeq.Class.jq_Method;
import joeq.Class.jq_Type;
import joeq.Compiler.Quad.ControlFlowGraph;
import joeq.Compiler.Quad.Operator;
import joeq.Compiler.Quad.Operator.Invoke.InvokeStatic;
import joeq.Compiler.Quad.Operator.MultiNewArray;
import joeq.Compiler.Quad.Operator.New;
import joeq.Compiler.Quad.Operator.NewArray;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.RegisterFactory.Register;
import chord.analyses.alias.Ctxt;
import chord.analyses.alias.DomC;
import chord.analyses.alloc.DomH;
import chord.analyses.argret.DomK;
import chord.analyses.invk.DomI;
import chord.analyses.method.DomM;
import chord.analyses.var.DomV;
import chord.bddbddb.Rel.RelView;
import chord.program.Program;
import chord.project.Chord;
import chord.project.ClassicProject;
import chord.project.Config;
import chord.project.analyses.JavaAnalysis;
import chord.project.analyses.ProgramRel;
import chord.util.ArraySet;
import chord.util.graph.IGraph;
import chord.util.graph.MutableGraph;
import chord.util.tuple.object.Pair;

/**
 * Analysis for pre-computing abstract contexts.
 * <p>
 * The goal of this analysis is to translate client-specified inputs concerning the desired kind of context sensitivity
 * into relations that are subsequently consumed by context-sensitive points-to and call-graph analyses.
 * <p>
 * This analysis allows:
 * <ul>
 *   <li>each method to be analyzed using a different kind of context sensitivity, namely, one of context insensitivity,
 *       k-CFA, k-object-sensitivity, and copy-context-sensitivity;</li>
 *   <li>each local variable to be analyzed context sensitively or insensitively; and</li>
 *   <li>a different 'k' value to be used for each object allocation site and method call site.</li>
 * </ul>
 * Recognized system properties:
 * <ul>
 *   <li>chord.inst.ctxt.kind: the kind of context sensitivity to use for each instance method (and all its locals).
 *       One of 'ci' (context insensitive), 'cs' (k-CFA), or 'co' (k-object-sensitive).  Default is 'ci'.</li>
 *   <li>chord.stat.ctxt.kind: the kind of context sensitivity to use for each static method (and all its locals).
 *       One of 'ci' (context insensitive), 'cs' (k-CFA), or 'co' (copy-context-sensitive).  Default is 'ci'.</li>
 *   <li>chord.ctxt.kind: the kind of context sensitivity to use for each method (and all its locals).
 *       One of 'ci', 'cs', or 'co'.  Serves as shorthand for properties chord.inst.ctxt.kind and chord.stat.ctxt.kind.</li>
 *   <li>chord.kobj.k and chord.kcfa.k: the 'k' value to use for each object allocation site and each method call site,
 *       respectively.  Default is 0.</li>
 * </ul>
 * <p>
 * This analysis outputs the following domains and relations:
 * <ul>
 *   <li>C: domain containing all abstract contexts</li>
 *   <li>CC: each (c,c2) such that c2 is all but the last element of context c</li>
 *   <li>CH: each (c,h) such that object allocation site h is the last element of abstract context c</li>
 *   <li>CI: each (c,i) such that call site i is the last element of abstract context c</li>
 *   <li>CVC: each (c,v,o) such that local v might point to object o in context c of its declaring method</li>
 *   <li>CFC: each (o1,f,o2) such that instance field f of object o1 might point to object o2</li>
 *   <li>FC: each (f,o) such that static field f may point to object o</li>
 *   <li>CICM: each (c,i,c2,m) if invocation i in context c can reach method m (in context c2)</li>
 *   <li>rootCM: each (c,m) such that method m is an entry method in context c</li>
 *   <li>reachableCM: each (c,m) such that method m can be called in context c</li>
 * </ul>
 *
 * @author Mayur Naik (mhn@cs.stanford.edu)
 */
@Chord(name = "simple-pro-ctxts-java",
       consumes = { "IM", "VH", "IK", "HK", "OK"},
       produces = { "C", "epsilonM", "kcfaSenM", "kobjSenM", "ctxtCpyM", "initCIC", "initCHC", "initCOC",
        "truncCKC", "roots", "CH", "CI", "CL" },
       namesOfTypes = { "C" },
       types = { DomC.class }
)
public class SimpleCtxtsAnalysis extends JavaAnalysis {
    private static final Set<Ctxt> emptyCtxtSet = Collections.emptySet();
    private static final Set<jq_Method> emptyMethSet = Collections.emptySet();
    private static final Quad[] emptyElems = new Quad[0];

    // includes all methods in domain
    private Set<Ctxt>[] methToCtxts;

    private TIntArrayList[] methToClrSites;  // ctxt kind is KCFASEN
    private TIntArrayList[] methToRcvSites;  // ctxt kind is KOBJSEN
    private Set<jq_Method>[] methToClrMeths; // ctxt kind is CTXTCPY

    private Set<jq_Method> roots;

    private Set<Ctxt> epsilonCtxtSet;

    public static final int CTXTINS = 0;  // abbr ci; must be 0
    public static final int KOBJSEN = 1;  // abbr co
    public static final int KCFASEN = 2;  // abbr cs
    public static final int CTXTCPY = 3;  // abbr cc

    private int[] ItoM;
    private int[] HtoM;
    private Quad[] ItoQ;
    private Quad[] HtoQ;

    private jq_Method mainMeth;
    private int[] methKind;       // indexed by domM
    private int[] kobjValue;      // indexed by domH
    private int[] kheapValue;      // indexed by domH
    private int[] kcfaValue;      // indexed by domI
    private int currHighestKcfaValue = 0;
    private int currHighestKobjValue = 0;
    private int currHighestKheapValue = 0;
    private int highestKcfaValue;
    private int highestKobjValue;
    private int highestKheapValue;

    private int instCtxtKind;
    private int statCtxtKind;

    private DomV domV;
    private DomM domM;
    private DomI domI;
    private DomH domH;
    private DomC domC;
    private DomK domK;

    private ProgramRel relIM;
    private ProgramRel relVH;
    private ProgramRel relIK;
    private ProgramRel relHK;
    private ProgramRel relOK;

    private ProgramRel relInitCIC;
    private ProgramRel relInitCHC;
    private ProgramRel relInitCOC;
    private ProgramRel relTruncCKC;
    private ProgramRel relRoots;
    private ProgramRel relCH;
    private ProgramRel relCI;
    private ProgramRel relCL;

    private ProgramRel relEpsilonM;
    private ProgramRel relKcfaSenM;
    private ProgramRel relKobjSenM;
    private ProgramRel relCtxtCpyM;

    public void run() {
        domV = (DomV) ClassicProject.g().getTrgt("V");
        domI = (DomI) ClassicProject.g().getTrgt("I");
        domM = (DomM) ClassicProject.g().getTrgt("M");
        domH = (DomH) ClassicProject.g().getTrgt("H");
        domC = (DomC) ClassicProject.g().getTrgt("C");
        domK = (DomK) ClassicProject.g().getTrgt("K");
        ClassicProject.g().runTask(domK);

        relIM = (ProgramRel) ClassicProject.g().getTrgt("IM");
        relVH = (ProgramRel) ClassicProject.g().getTrgt("VH");
        relIK = (ProgramRel) ClassicProject.g().getTrgt("IK");
        relHK = (ProgramRel) ClassicProject.g().getTrgt("HK");
        relOK = (ProgramRel) ClassicProject.g().getTrgt("OK");

        relInitCIC = (ProgramRel) ClassicProject.g().getTrgt("initCIC");
        relInitCHC = (ProgramRel) ClassicProject.g().getTrgt("initCHC");
        relInitCOC = (ProgramRel) ClassicProject.g().getTrgt("initCOC");
        relTruncCKC = (ProgramRel) ClassicProject.g().getTrgt("truncCKC");
        relRoots = (ProgramRel) ClassicProject.g().getTrgt("roots");
        relCH = (ProgramRel) ClassicProject.g().getTrgt("CH");
        relCI = (ProgramRel) ClassicProject.g().getTrgt("CI");
        relCL = (ProgramRel) ClassicProject.g().getTrgt("CL");

        relEpsilonM = (ProgramRel) ClassicProject.g().getTrgt("epsilonM");
        relKcfaSenM = (ProgramRel) ClassicProject.g().getTrgt("kcfaSenM");
        relKobjSenM = (ProgramRel) ClassicProject.g().getTrgt("kobjSenM");
        relCtxtCpyM = (ProgramRel) ClassicProject.g().getTrgt("ctxtCpyM");

        mainMeth = Program.g().getMainMethod();

        String ctxtKindStr = System.getProperty("chord.ctxt.kind", "ci");
        Config.check(ctxtKindStr, new String[] { "ci", "cs", "co" }, "chord.ctxt.kind");
        String instCtxtKindStr = System.getProperty("chord.inst.ctxt.kind", ctxtKindStr);
        Config.check(instCtxtKindStr, new String[] { "ci", "cs", "co" }, "chord.inst.ctxt.kind");
        String statCtxtKindStr = System.getProperty("chord.stat.ctxt.kind", ctxtKindStr);
        Config.check(statCtxtKindStr, new String[] { "ci", "cs", "co" }, "chord.stat.ctxt.kind");
        if (instCtxtKindStr.equals("ci")) {
            instCtxtKind = CTXTINS;
        } else if (instCtxtKindStr.equals("cs")) {
            instCtxtKind = KCFASEN;
        } else
            instCtxtKind = KOBJSEN;
        if (statCtxtKindStr.equals("ci")) {
            statCtxtKind = CTXTINS;
        } else if (statCtxtKindStr.equals("cs")) {
            statCtxtKind = KCFASEN;
        } else
            statCtxtKind = CTXTCPY;

        int kobjK = Integer.getInteger("chord.kobj.k", 0);
        currHighestKobjValue = kobjK;
        int kheapK = Integer.getInteger("chord.kheap.k", 1);
        currHighestKheapValue = kheapK;
        //assert (kobjK > 0);
        int kcfaK = Integer.getInteger("chord.kcfa.k", 0);
        currHighestKcfaValue = kcfaK;
        // assert (kobjK <= kcfaK+1)

        highestKobjValue = Integer.getInteger("chord.kobj.khighest", 100);
        //assert (highestKobjValue > 0);
        highestKheapValue = Integer.getInteger("chord.kheap.khighest", 100);
        highestKcfaValue = Integer.getInteger("chord.kcfa.khighest", 100);

        int numV = domV.size();
        int numM = domM.size();
        int numA = domH.getLastI() + 1;
        int numI = domI.size();

        {
            // set k values to use for sites: build arrays kobjValue and kcfaValue

            kobjValue = new int[numA];
            kheapValue = new int[numA];
            HtoM = new int[numA]; // Which method is h located in?
            HtoQ = new Quad[numA];
            for (int i = 1; i < numA; i++) {
                kobjValue[i] = kobjK;
                kheapValue[i] = kheapK;
                Quad site = (Quad) domH.get(i);
                jq_Method m = site.getMethod();
                HtoM[i] = domM.indexOf(m);
                HtoQ[i] = site;
            }

            kcfaValue = new int[numI];
            ItoM = new int[numI]; // Which method is i located in?
            ItoQ = new Quad[numI];
            for (int i = 0; i < numI; i++) {
                kcfaValue[i] = kcfaK;
                Quad invk = domI.get(i);
                jq_Method m = invk.getMethod();
                ItoM[i] = domM.indexOf(m);
                ItoQ[i] = invk;
            }

            relIK.load();
            Iterable<Pair<Quad, Integer>> tuplesIK = relIK.getAry2ValTuples();
            for (Pair<Quad, Integer> t : tuplesIK){
                kcfaValue[domI.indexOf(t.val0)] = t.val1;
                currHighestKcfaValue = currHighestKcfaValue < t.val1 ? t.val1 : currHighestKcfaValue;
            }
            relIK.close();

            relHK.load();
            Iterable<Pair<Quad, Integer>> tuplesHK = relHK.getAry2ValTuples();
            for (Pair<Quad, Integer> t : tuplesHK){
                kheapValue[domH.indexOf(t.val0)] = t.val1;
                currHighestKheapValue = currHighestKheapValue < t.val1 ? t.val1 : currHighestKheapValue;
            }
            relHK.close();

            relOK.load();
            Iterable<Pair<Quad, Integer>> tuplesOK = relOK.getAry2ValTuples();
            for (Pair<Quad, Integer> t : tuplesOK){
                kobjValue[domH.indexOf(t.val0)] = t.val1;
                currHighestKobjValue = currHighestKobjValue < t.val1 ? t.val1 : currHighestKobjValue;
            }
            relOK.close();
        }

        {
            // populate arrays methKind

            // set the kind of context sensitivity to use for various methods
            methKind = new int[numM];
            for (int mIdx = 0; mIdx < numM; mIdx++) {
                jq_Method m = domM.get(mIdx);
                int kind;
                if (m == mainMeth || m instanceof jq_ClassInitializer || m.isAbstract())
                    kind = CTXTINS;
                else
                    kind = m.isStatic() ? statCtxtKind : instCtxtKind;
                methKind[mIdx] = kind;
            }

        }

        validate();

        relIM.load();
        relVH.load();

        Ctxt epsilon = domC.setCtxt(emptyElems);
        epsilonCtxtSet = new ArraySet<Ctxt>(1);
        epsilonCtxtSet.add(epsilon);

        methToCtxts = new Set[numM];

        methToClrSites = new TIntArrayList[numM];
        methToRcvSites = new TIntArrayList[numM];
        methToClrMeths = new Set[numM];
        roots = new HashSet<jq_Method>();

        System.out.println("DomC Size:" + domC.size());
        // Do the heavy crunching
        doAnalysis();

        relIM.close();
        relVH.close();

        // Populate domC
        for (int iIdx = 0; iIdx < numI; iIdx++) {
            Quad invk = (Quad) domI.get(iIdx);
            jq_Method meth = invk.getMethod();
            int mIdx = domM.indexOf(meth);
            Set<Ctxt> ctxts = methToCtxts[mIdx];
            final int kLim = Math.min(kcfaValue[iIdx] + 1, highestKcfaValue);
            for (int j = 0; j <= kLim; j++) {
                for (Ctxt oldCtxt : ctxts) {
                    Quad[] oldElems = oldCtxt.getElems();
                    Quad[] newElems = combine(j, invk, oldElems);
                    domC.setCtxt(newElems);
                }
            }
        }
        for (int hIdx = 1; hIdx < numA; hIdx++) {
            Quad inst = (Quad) domH.get(hIdx);
            jq_Method meth = inst.getMethod();
            int mIdx = domM.indexOf(meth);
            Set<Ctxt> ctxts = methToCtxts[mIdx];
            final int kOLim = Math.min(kobjValue[hIdx] + 1, highestKobjValue);
            final int kHLim = Math.min(kheapValue[hIdx] + 1, highestKheapValue);
            final int kLim = Math.max(kOLim, kHLim);
            for(int j = 0; j <= kLim; j++){
                for (Ctxt oldCtxt : ctxts) {
                    Quad[] oldElems = oldCtxt.getElems();
                    Quad[] newElems = combine(j, inst, oldElems);
                    domC.setCtxt(newElems);
                }
            }
        }


        domC.save();

        int numC = domC.size();

        relCL.zero();
        for (int cIdx = 0; cIdx < numC; cIdx++) {
            Ctxt ctxt = domC.get(cIdx);
            relCL.add(ctxt, new Integer(ctxt.length()));
        }
        relCL.save();

        assert (domC.size() == numC);

        relCI.zero();
        relInitCIC.zero();
        for (int iIdx = 0; iIdx < numI; iIdx++) {
            Quad invk = (Quad) domI.get(iIdx);
            jq_Method meth = invk.getMethod();
            Set<Ctxt> ctxts = methToCtxts[domM.indexOf(meth)];
            final int kLim = Math.min(kcfaValue[iIdx] + 1, highestKcfaValue);
            for (Ctxt oldCtxt : ctxts) {
                Quad[] oldElems = oldCtxt.getElems();
                Quad[] newElems = combine(kLim, invk, oldElems);
                Ctxt newCtxt = domC.setCtxt(newElems);
                relInitCIC.add(oldCtxt, invk, newCtxt);
                relCI.add(newCtxt, invk);
                if (kcfaValue[iIdx] == 0) {
                    newElems = combine(0, invk, oldElems);
                    newCtxt = domC.setCtxt(newElems);
                    relCI.add(newCtxt, invk);
                }
            }
        }
        relInitCIC.save();
        relCI.save();

        assert (domC.size() == numC);


        relCH.zero();
        relInitCOC.zero();
        relInitCHC.zero();
        for (int hIdx = 1; hIdx < numA; hIdx++) {
            Quad inst = (Quad) domH.get(hIdx);
            jq_Method meth = inst.getMethod();
            int mIdx = domM.indexOf(meth);
            Set<Ctxt> ctxts = methToCtxts[mIdx];
            final int kOLim = Math.min(kobjValue[hIdx] + 1, highestKobjValue);
            final int kHLim = Math.min(kheapValue[hIdx] + 1, highestKheapValue);
            final int kLim = Math.max(kOLim, kHLim);
            // XXX(rgrig): I'm not sure why using distinct kOLim and kHLim
            // makes the analysis nonmonotonic (in the queries derived). Fix?
            for (Ctxt oldCtxt : ctxts) {
                Quad[] oldElems = oldCtxt.getElems();
                Quad[] newElems = combine(/*kOLim*/ kLim, inst, oldElems);
                Ctxt newCtxt = domC.setCtxt(newElems);
                relInitCOC.add(oldCtxt, inst, newCtxt);
                relCH.add(newCtxt, inst);
                newElems = combine(/*kHLim*/ kLim, inst, oldElems);
                newCtxt = domC.setCtxt(newElems);
                relInitCHC.add(oldCtxt, inst, newCtxt);
                relCH.add(newCtxt, inst);

                if (kobjValue[hIdx] == 0 || kheapValue[hIdx] == 0) {
                    newElems = combine(0, inst, oldElems);
                    newCtxt = domC.setCtxt(newElems);
                    relCH.add(newCtxt, inst);
                }
            }
        }
        relInitCOC.save();
        relInitCHC.save();
        relCH.save();

        assert (domC.size() == numC);


        int currhighestKValue = (currHighestKcfaValue < currHighestKobjValue) ? currHighestKobjValue : currHighestKcfaValue;
        currhighestKValue = (currhighestKValue < currHighestKheapValue) ? currHighestKheapValue : currhighestKValue;

        relTruncCKC.zero();
        for (int cIdx = 0; cIdx < numC; cIdx++) {
            Ctxt fullCtxt = domC.get(cIdx);
            Quad[] allElems = fullCtxt.getElems();
            for(int z = 0; z <= currhighestKValue+1; z++){
            //    Quad[] truncElems = truncate(z, allElems);
                Quad[] truncElems = truncate(z==0?z:z-1, allElems);
                Ctxt truncCtxt = domC.setCtxt(truncElems);
                relTruncCKC.add(fullCtxt, new Integer(z), truncCtxt);
            }

        }
        relTruncCKC.save();

        assert (domC.size() == numC);

        relEpsilonM.zero();
        relKcfaSenM.zero();
        relKobjSenM.zero();
        relCtxtCpyM.zero();
        for (int mIdx = 0; mIdx < numM; mIdx++) {
            int kind = methKind[mIdx];
            switch (kind) {
            case CTXTINS:
                relEpsilonM.add(mIdx);
                break;
            case KOBJSEN:
                relKobjSenM.add(mIdx);
                break;
            case KCFASEN:
                relKcfaSenM.add(mIdx);
                break;
            case CTXTCPY:
                relCtxtCpyM.add(mIdx);
                break;
            default:
                assert false;
            }
        }
        relEpsilonM.save();
        relKcfaSenM.save();
        relKobjSenM.save();
        relCtxtCpyM.save();

        relRoots.zero();
        for(jq_Method m : roots){
            relRoots.add(m);
        }
        relRoots.save();

    }

    private void validate() {
        // check that the main jq_Method and each class initializer method and each method without a body
        // is not asked to be analyzed context sensitively.
        int numM = domM.size();
        for (int m = 0; m < numM; m++) {
            int kind = methKind[m];
            if (kind != CTXTINS) {
                jq_Method meth = domM.get(m);
                assert (meth != mainMeth);
                assert (!(meth instanceof jq_ClassInitializer));
                if (kind == KOBJSEN) {
                    assert (!meth.isStatic());
                } else if (kind == CTXTCPY) {
                    assert (meth.isStatic());
                }
            }
        }
    }

    private void doAnalysis() {
        Map<jq_Method, Set<jq_Method>> methToPredsMap = new HashMap<jq_Method, Set<jq_Method>>();
        for (int mIdx = 0; mIdx < domM.size(); mIdx++) { // For each method...
            jq_Method meth = domM.get(mIdx);
            int kind = methKind[mIdx];
            switch (kind) {
            case CTXTINS:
            {
                roots.add(meth);
                methToPredsMap.put(meth, emptyMethSet);
                methToCtxts[mIdx] = epsilonCtxtSet;
                break;
            }
            case KCFASEN:
            {
                Set<jq_Method> predMeths = new HashSet<jq_Method>();
                TIntArrayList clrSites = new TIntArrayList();
                for (Quad invk : getCallers(meth)) {
                    predMeths.add(invk.getMethod()); // Which method can point to this method...?
                    int iIdx = domI.indexOf(invk);
                    clrSites.add(iIdx); // sites that can call me
                }
                methToClrSites[mIdx] = clrSites;
                methToPredsMap.put(meth, predMeths);
                methToCtxts[mIdx] = emptyCtxtSet;
                break;
            }
            case KOBJSEN:
            {
                Set<jq_Method> predMeths = new HashSet<jq_Method>();
                TIntArrayList rcvSites = new TIntArrayList();
                ControlFlowGraph cfg = meth.getCFG();
                Register thisVar = cfg.getRegisterFactory().get(0);
                Iterable<Quad> pts = getPointsTo(thisVar);
                for (Quad inst : pts) {
                    predMeths.add(inst.getMethod());
                    int hIdx = domH.indexOf(inst);
                    rcvSites.add(hIdx);
                }
                methToRcvSites[mIdx] = rcvSites;
                methToPredsMap.put(meth, predMeths);
                methToCtxts[mIdx] = emptyCtxtSet;
                break;
            }
            case CTXTCPY:
            {
                Set<jq_Method> predMeths = new HashSet<jq_Method>();
                for (Quad invk : getCallers(meth)) {
                    predMeths.add(invk.getMethod());
                }
                methToClrMeths[mIdx] = predMeths;
                methToPredsMap.put(meth, predMeths);
                methToCtxts[mIdx] = emptyCtxtSet;
                break;
            }
            default:
                assert false;
            }
        }
        process(roots, methToPredsMap);
    }

    // Compute all the contexts that each method can be called in
    private void process(Set<jq_Method> roots, Map<jq_Method, Set<jq_Method>> methToPredsMap) {
        IGraph<jq_Method> graph = new MutableGraph<jq_Method>(roots, methToPredsMap, null);
        List<Set<jq_Method>> sccList = graph.getTopSortedSCCs();
        int n = sccList.size();
        if (Config.verbose >= 2)
            System.out.println("numSCCs: " + n);
        for (int i = 0; i < n; i++) { // For each SCC...
            Set<jq_Method> scc = sccList.get(i);
            if (Config.verbose >= 2)
                System.out.println("Processing SCC #" + i + " of size: " + scc.size());
            if (scc.size() == 1) { // Singleton
                jq_Method cle = scc.iterator().next();
                if (roots.contains(cle))
                    continue;
                if (!graph.hasEdge(cle, cle)) {
                    int cleIdx = domM.indexOf(cle);
                    methToCtxts[cleIdx] = getNewCtxts(cleIdx);
                    continue;
                }
            }
            for (jq_Method cle : scc) {
                assert (!roots.contains(cle));
            }
            boolean changed = true;
            for (int count = 0; changed; count++) { // Iterate...
                if (Config.verbose >= 2)
                    System.out.println("\tIteration  #" + count);
                changed = false;
                for (jq_Method cle : scc) { // For each node (method) in SCC
                    int mIdx = domM.indexOf(cle);
                    Set<Ctxt> newCtxts = getNewCtxts(mIdx);
                    if (!changed) {
                        Set<Ctxt> oldCtxts = methToCtxts[mIdx];
                        if (newCtxts.size() > oldCtxts.size())
                            changed = true;
                        else {
                            for (Ctxt ctxt : newCtxts) {
                                if (!oldCtxts.contains(ctxt)) {
                                    changed = true;
                                    break;
                                }
                            }
                        }
                    }
                    methToCtxts[mIdx] = newCtxts;
                }
            }
        }
    }

    private Iterable<Quad> getPointsTo(Register var) {
        RelView view = relVH.getView();
        view.selectAndDelete(0, var);
        return view.getAry1ValTuples();
    }

    private Iterable<Quad> getCallers(jq_Method meth) {
        RelView view = relIM.getView();
        view.selectAndDelete(1, meth);
        return view.getAry1ValTuples();
    }

    private Quad[] combine(int k, Quad inst, Quad[] elems) {
        int oldLen = elems.length;
        int newLen = Math.min(k - 1, oldLen) + 1;
        Quad[] newElems = new Quad[newLen];
        if (newLen > 0) newElems[0] = inst;
        if (newLen > 1)
            System.arraycopy(elems, 0, newElems, 1, newLen - 1);
        return newElems;
    }

    private Quad[] truncate(int z, Quad[] elems) {
        int oldLen = elems.length;
        int newLen = Math.min(z, oldLen);
        Quad[] newElems = new Quad[newLen];
        if (newLen > 0)
            System.arraycopy(elems, 0, newElems, 0, newLen);

        return newElems;
    }

    private Set<Ctxt> getNewCtxts(int cleIdx) { // Update contexts for this method (callee)
        final Set<Ctxt> newCtxts = new HashSet<Ctxt>();
        System.out.printf("YES %d called%n", cleIdx);
        int kind = methKind[cleIdx];
        switch (kind) {
        case KCFASEN:
        {
            TIntArrayList invks = methToClrSites[cleIdx]; // which call sites point to me
            int n = invks.size();
            for (int i = 0; i < n; i++) {
                int iIdx = invks.get(i);
                Quad invk = ItoQ[iIdx];
                int k = kcfaValue[iIdx];
                int clrIdx = ItoM[iIdx];
                Set<Ctxt> clrCtxts = methToCtxts[clrIdx]; // method of caller
                for(int j = 0; j <= k; j++){
                    for (Ctxt oldCtxt : clrCtxts) {
                        Quad[] oldElems = oldCtxt.getElems();
                        Quad[] newElems = combine(j, invk, oldElems); // Append
                        Ctxt newCtxt = domC.setCtxt(newElems);
                        newCtxts.add(newCtxt);
                    }
                }
            }
            break;
        }
        case KOBJSEN:
        {
            TIntArrayList rcvs = methToRcvSites[cleIdx];
            int n = rcvs.size();
            if (cleIdx == 1260) System.out.printf("YES n=%d%n", n);
            for (int i = 0; i < n; i++) {
                int hIdx = rcvs.get(i);
                Quad rcv = HtoQ[hIdx];
                int k = Math.max(kobjValue[hIdx], kheapValue[hIdx]);
                if (cleIdx == 1260) System.out.printf("YES hIdx=%d k=%d%n", hIdx, k);
                int clrIdx = HtoM[hIdx];
                Set<Ctxt> rcvCtxts = methToCtxts[clrIdx];
                for(int j = 0; j <= k; j++){
                    for (Ctxt oldCtxt : rcvCtxts) {
                        Quad[] oldElems = oldCtxt.getElems();
                        Quad[] newElems = combine(j, rcv, oldElems);
                        Ctxt newCtxt = domC.setCtxt(newElems);
                        newCtxts.add(newCtxt);
                    }
                }
            }
            break;
        }
        case CTXTCPY:
        {
            Set<jq_Method> clrs = methToClrMeths[cleIdx];
            for (jq_Method clr : clrs) {
                int clrIdx = domM.indexOf(clr);
                Set<Ctxt> clrCtxts = methToCtxts[clrIdx];
                newCtxts.addAll(clrCtxts);
            }
            break;
        }
        default:
            assert false;
        }
        return newCtxts;
    }

    public static String getCspaKind() {
        String ctxtKindStr = System.getProperty("chord.ctxt.kind", "ci");
        String instCtxtKindStr = System.getProperty("chord.inst.ctxt.kind", ctxtKindStr);
        String statCtxtKindStr = System.getProperty("chord.stat.ctxt.kind", ctxtKindStr);
        int instCtxtKind, statCtxtKind;
        if (instCtxtKindStr.equals("ci")) {
            instCtxtKind = SimpleCtxtsAnalysis.CTXTINS;
        } else if (instCtxtKindStr.equals("cs")) {
            instCtxtKind = SimpleCtxtsAnalysis.KCFASEN;
        } else if (instCtxtKindStr.equals("co")) {
            instCtxtKind = SimpleCtxtsAnalysis.KOBJSEN;
        } else
            throw new RuntimeException();
        if (statCtxtKindStr.equals("ci")) {
            statCtxtKind = SimpleCtxtsAnalysis.CTXTINS;
        } else if (statCtxtKindStr.equals("cs")) {
            statCtxtKind = SimpleCtxtsAnalysis.KCFASEN;
        } else if (statCtxtKindStr.equals("co")) {
            statCtxtKind = SimpleCtxtsAnalysis.CTXTCPY;
        } else
            throw new RuntimeException();
        String cspaKind;
        if (instCtxtKind == SimpleCtxtsAnalysis.CTXTINS && statCtxtKind == SimpleCtxtsAnalysis.CTXTINS)
            cspaKind = "cspa-0cfa-dlog";
        else if (instCtxtKind == SimpleCtxtsAnalysis.KOBJSEN && statCtxtKind == SimpleCtxtsAnalysis.CTXTCPY)
            cspaKind = "cspa-kobj-dlog";
        else if (instCtxtKind == SimpleCtxtsAnalysis.KCFASEN && statCtxtKind == SimpleCtxtsAnalysis.KCFASEN)
            cspaKind = "cspa-kcfa-dlog";
        else
            cspaKind = "cspa-hybrid-dlog";
        return cspaKind;
    }

    jq_Type h2t(Quad h) {
        Operator op = h.getOperator();
        if (op instanceof New)
            return New.getType(h).getType();
        else if (op instanceof NewArray)
            return NewArray.getType(h).getType();
        else if (op instanceof MultiNewArray)
            return MultiNewArray.getType(h).getType();
        else
            return null;
    }
    String hstr(Quad h) {
        String path = new File(h.toJavaLocStr()).getName();
        jq_Type t = h2t(h);
        return path+"("+(t == null ? "?" : t.shortName())+")";
    }
    String istr(Quad i) {
        String path = new File(i.toJavaLocStr()).getName();
        jq_Method m = InvokeStatic.getMethod(i).getMethod();
        return path+"("+m.getName()+")";
    }
    String jstr(Quad j) { return isAlloc(j) ? hstr(j) : istr(j); }
    String estr(Quad e) {
        String path = new File(e.toJavaLocStr()).getName();
        Operator op = e.getOperator();
        return path+"("+op+")";
    }
    String cstr(Ctxt c) {
        StringBuilder buf = new StringBuilder();
        buf.append('{');
        for (int i = 0; i < c.length(); i++) {
            if (i > 0) buf.append(" | ");
            Quad q = c.get(i);
            buf.append(isAlloc(q) ? hstr(q) : istr(q));
        }
        buf.append('}');
        return buf.toString();
    }
    String fstr(jq_Field f) { return f.getDeclaringClass()+"."+f.getName(); }
    String vstr(Register v) { return v+"@"+mstr(domV.getMethod(v)); }
    String mstr(jq_Method m) { return m.getDeclaringClass().shortName()+"."+m.getName(); }
    boolean isAlloc(Quad q) { return domH.indexOf(q) != -1; }
}
