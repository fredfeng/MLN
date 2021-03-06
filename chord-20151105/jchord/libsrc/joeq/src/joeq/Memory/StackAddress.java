// StackAddress.java, created Wed Sep 18  1:22:46 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Memory;

import joeq.Class.PrimordialClassLoader;
import joeq.Class.jq_Class;
import joeq.Class.jq_StaticField;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id: StackAddress.java,v 1.7 2004/03/09 21:56:18 jwhaley Exp $
 */
public class StackAddress extends Address {

    public static StackAddressFactory FACTORY;
    
    public abstract static class StackAddressFactory {
        public abstract int size();
        public abstract StackAddress getBasePointer();
        public abstract StackAddress getStackPointer();
        public abstract StackAddress alloca(int size);
    }
    
    public static final int size() {
        return FACTORY.size();
    }
    public static final StackAddress getBasePointer() {
        return FACTORY.getBasePointer();
    }
    public static final StackAddress getStackPointer() {
        return FACTORY.getStackPointer();
    }
    public static final StackAddress alloca(int size) {
        return FACTORY.alloca(size);
    }
    
    public native Address peek();
    public native byte    peek1();
    public native short   peek2();
    public native int     peek4();
    public native long    peek8();
    
    public native void poke(Address v);
    public native void poke1(byte v);
    public native void poke2(short v);
    public native void poke4(int v);
    public native void poke8(long v);
    
    public native Address offset(int offset);
    public native Address align(int shift);
    public native int difference(Address v);
    public native boolean isNull();
    
    public native int to32BitValue();
    public native String stringRep();
    
    public static final jq_Class _class;
    public static final jq_StaticField _FACTORY;
    static {
        _class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Memory/StackAddress;");
        _FACTORY = _class.getOrCreateStaticField("FACTORY", "Ljoeq/Memory/StackAddress$StackAddressFactory;");
    }
}
