package chord.analyses.confdep.rels;

import joeq.Class.jq_Class;
import joeq.Class.jq_Method;
import joeq.Compiler.Quad.Operand;
import joeq.Compiler.Quad.Quad;
import joeq.Compiler.Quad.Operator.Move;
import joeq.Compiler.Quad.RegisterFactory.Register;
import chord.analyses.var.DomV;
import chord.program.visitors.*;
import chord.project.Chord;
import chord.project.analyses.ProgramRel;


/**
 * Contains points (p0,v0) where at point p, value v is set to null.
 * @author asrabkin
 *
 */
@Chord(
		name = "nullConst",
		sign = "P0,V0:P0_V0"
)
public class RelNullConsts extends ProgramRel implements IMoveInstVisitor {


	public void visit(jq_Class c) { }


	public void visitMoveInst(Quad q) {
		Operand operand = Move.getSrc(q);
		Register dest = Move.getDest(q).getRegister();

		if (operand instanceof Operand.AConstOperand) {
			Object wrapped = ((Operand.AConstOperand) operand).getWrapped();
			if (wrapped == null)
				super.add(q, dest);
		}
	}

	@Override
	public void visit(jq_Method m) {
	}


}
