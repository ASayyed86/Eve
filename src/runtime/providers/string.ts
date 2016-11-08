//---------------------------------------------------------------------
// String providers
//---------------------------------------------------------------------

import {Constraint} from "../join";
import * as providers from "./index";

// Concat strings together. Args expects a set of variables/string constants
// to concatenate together and an array with a single return variable
class Concat extends Constraint {
  // To resolve a proposal, we concatenate our resolved args
  resolveProposal(proposal, prefix) {
    let {args} = this.resolve(prefix);
    return [args.join("")];
  }

  // We accept a prefix if the return is equivalent to concatentating
  // all the args
  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    return args.join("") === returns[0];
  }

  // concat always returns cardinality 1
  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    proposal.providing = proposed;
    proposal.cardinality = 1;
    return proposal;
  }
}



class Split extends Constraint {
  static AttributeMapping = {
    "text": 0,
    "by": 1,
  }
  static ReturnMapping = {
    "token": 0,
    "index": 1,
  }

  returnType: "both" | "index" | "token";

  constructor(id: string, args: any[], returns: any[]) {
    super(id, args, returns);
    if(this.returns[1] !== undefined && this.returns[0] !== undefined) {
      this.returnType = "both"
    } else if(this.returns[1] !== undefined) {
      this.returnType = "index";
    } else {
      this.returnType = "token";
    }
  }

  resolveProposal(proposal, prefix) {
    let {returns} = this.resolve(prefix);
    let tokens = proposal.index;
    let results = tokens;
    if(this.returnType === "both") {
      results = [];
      let ix = 1;
      for(let token of tokens) {
        results.push([token, ix]);
        ix++;
      }
    } else if(this.returnType === "index") {
      results = [];
      let ix = 1;
      for(let token of tokens) {
        results.push(ix);
        ix++;
      }
    }
    return results;
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    // @TODO: this is expensive, we should probably try to cache the split somehow
    return args[0].split(args[1])[returns[1]] === returns[0];
  }

  getProposal(tripleIndex, proposed, prefix) {
    let {args} = this.resolve(prefix);
    let proposal = this.proposalObject;
    if(this.returnType === "both") {
      proposal.providing = [this.returns[0], this.returns[1]];
    } else if(this.returnType == "index") {
      proposal.providing = this.returns[1];
    } else {
      proposal.providing = this.returns[0];
    }
    proposal.index = args[0].split(args[1]);
    proposal.cardinality = proposal.index.length;
    return proposal;
  }
}


class Replace extends Constraint {
  static AttributeMapping = {
    "text": 0,
    "subtext": 1,
    "with": 2,
  }

  resolveProposal(proposal, prefix) {
    let {args, returns} = this.resolve(prefix);
    let [text, subtext, _with] = args; // with is a reserved word
    return [text.replace(subtext, _with)];
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let [text, subtext, _with] = args;
    if(typeof text !== "string") return false;
    return text.replace(subtext, _with) === returns[0];
  }

  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args} = this.resolve(prefix);
    if(typeof args[0] !== "string") {
      proposal.cardinality = 0;
    } else {
      proposal.providing = proposed;
      proposal.cardinality = 1;
    }
    return proposal;
  }
}

class Length extends Constraint {
  static AttributeMapping = {
    "text": 0,
  }
  resolveProposal(proposal, prefix) {
    let {args} = this.resolve(prefix);
    let [text] = args;
    return [text.length];
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let text = args[0];
    if(typeof text !== "string") return false;
    return text.length === returns[0];
  }

  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args} = this.resolve(prefix);
    if(typeof args[0] !== "string") {
      proposal.cardinality = 0;
    } else {
      proposal.providing = proposed;
      proposal.cardinality = 1;
    }
    return proposal;
  }
}


class CharAt extends Constraint {
  static AttributeMapping = {
    "text": 0,
    "index": 1,
  }
  resolveProposal(proposal, prefix) {
    let {args} = this.resolve(prefix);
    let [text, index] = args;
    return [text[index]];
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let [text, index] = args;
    if(typeof text !== "string") return false;
    if(index < 0 || index >= text.length) return false;
    return text[index] === returns[0];
  }

  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args, returns} = this.resolve(prefix);
    let [text, index] = args;
    if(typeof text !== "string" || index < 0 || index >= text.length) {
      proposal.cardinality = 0;
    } else {
      proposal.providing = proposed;
      proposal.cardinality = 1;
    }
    return proposal;
  }
}


class Find extends Constraint {
  static AttributeMapping = {
    "text": 0,
    "subtext": 1,
    "case-sensitive": 2,
  }

  findAll(text, subtext, caseSensitive) {
    text = caseSensitive ? text.toLowerCase() : text;
    subtext = caseSensitive ? subtext.toLowerCase() : subtext;
    let temp = [];
    for (let i = 0; i < text.length; i ++) {
      if (text.substring(i, i + subtext.length) === subtext) {
         temp.push(i);
      }
    }
    return temp;
  }

  testAll(text, subtext, indexes, caseSensitive) {
    text = caseSensitive ? text.toLowerCase() : text;
    subtext = caseSensitive ? subtext.toLowerCase() : subtext;
    for (let i of indexes) {
      if (i < 0 || i > text.length || text.substring(i, i + subtext.length) !== subtext) {
        return false;
      }
    }
    return true;
  }

  resolveProposal(proposal, prefix) {
    let {args} = this.resolve(prefix);
    let [text, subtext, caseSensitive=true] = args;
    return this.findAll(text, subtext, caseSensitive);
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let [text, subtext, caseSensitive=true] = args;
    console.log('test', returns)
    let [indexes] = returns;
    if(typeof text !== "string") return false;
    return this.testAll(text, subtext, indexes, caseSensitive);
  }

  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args,} = this.resolve(prefix);
    let [text, subtext, caseSensitive=true] = args;
    if(typeof text !== "string") {
      proposal.cardinality = 0;
    } else {
      proposal.providing = proposed;
      proposal.cardinality = this.findAll(text, subtext, caseSensitive).length;
    }
    return proposal;
  }
}

// substring over the field 'text', with the base index being 1, inclusive, 'from' defaulting
// to the beginning of the string, and 'to' the end
class Substring extends Constraint {
  static AttributeMapping = {
    "text": 0,
    "from": 1,
    "to": 2,
  }
  static ReturnMapping = {
    "value": 0,
  }
  // To resolve a proposal, we concatenate our resolved args
  resolveProposal(proposal, prefix) {
    let {args, returns} = this.resolve(prefix);
    let from = 0;
    let text = args[0];
    let to = text.length;
    if (args[1] != undefined) from = args[1] - 1;
    if (args[2] != undefined) to = args[2];
    return [text.substring(from, to)];
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let from = 0;
    let text = args[0];
    if(typeof text !== "string") return false;
    let to = text.length;
    if (args[1] != undefined) from = args[1] - 1;
    if (args[2] != undefined) to = args[2];
    console.log("test string", text.substring(from, to), from, to, returns[0]);
    return text.substring(from, to) === returns[0];
  }

  // substring always returns cardinality 1
  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args} = this.resolve(prefix);
    if(typeof args[0] !== "string") {
      proposal.cardinality = 0;
    } else {
      proposal.providing = proposed;
      proposal.cardinality = 1;
    }
    return proposal;
  }
}

class Convert extends Constraint {
  static AttributeMapping = {
    "value": 0,
    "to": 1,
  }
  static ReturnMapping = {
    "converted": 0,
  }

  resolveProposal(proposal, prefix) {
    let {args, returns} = this.resolve(prefix);
    let from = 0;
    let value = args[0];
    let to = args[1];
    let converted;
    if(to === "number") {
      converted = +value;
      if(isNaN(converted)) throw new Error("Unable to deal with NaN in the proposal stage.");
    } else if(to === "string") {
      converted = ""+value;
    }
    return [converted];
  }

  test(prefix) {
    let {args, returns} = this.resolve(prefix);
    let value = args[0];
    let to = args[1];

    let converted;
    if(to === "number") {
      converted = +value;
      if(isNaN(converted)) return false;
      if(converted === "") return false;
      return
    } else if(to === "string") {
      converted = ""+value;
    } else {
      return false;
    }

    return converted === returns[0];
  }

  // 1 if valid, 0 otherwise
  getProposal(tripleIndex, proposed, prefix) {
    let proposal = this.proposalObject;
    let {args} = this.resolve(prefix);
    let value = args[0];
    let to = args[1];

    proposal.cardinality = 1;
    proposal.providing = proposed;

    if(to === "number") {
      if(isNaN(+value) || value === "") proposal.cardinality = 0;
    } else if(to === "string") {
    } else {
      proposal.cardinality = 0;
    }

    return proposal;
  }
}

providers.provide("find", Find);
providers.provide("char-at", CharAt);
providers.provide("replace", Replace);
providers.provide("length", Length);
providers.provide("concat", Concat);
providers.provide("split", Split);
providers.provide("substring", Substring);
providers.provide("convert", Convert);
