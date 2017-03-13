//--------------------------------------------------------------------
// The Eve compiler as a watcher
//--------------------------------------------------------------------

import {Watcher, RawValue, DiffConsumer} from "./watcher";
import {ID, Block} from "../runtime/runtime";
import {LinearFlow, ReferenceContext, Reference, Record, Value, WatchFlow} from "../runtime/dsl2";
import "setimmediate";

var watcherFunctions:{[name:string]: DiffConsumer} = {};

export function registerWatcherFunction(name:string, consumer:DiffConsumer) {
  watcherFunctions[name] = consumer;
}

interface CompilationContext {
  variables: {[id:string]: Reference},
}

class CompilerWatcher extends Watcher {

  blocksToCompile:{[blockID:string]: boolean} = {};
  blocksToRemove:{[blockID:string]: boolean} = {};
  blocks:{[blockID:string]: Block} = {};
  items:{[id:string]: any} = {};

  //------------------------------------------------------------------
  // Compile queue
  //------------------------------------------------------------------

  queued = false;
  queue(blockID:string, isAdd = true) {
    if(isAdd) this.blocksToCompile[blockID] = true;
    this.blocksToRemove[blockID] = true;

    if(!this.queued) {
      this.queued = true;
      setImmediate(this.runQueue)
    }
  }
  runQueue = () => {
    let adds = [];
    let removes = [];
    for(let ID in this.blocksToRemove) {
      if(!this.blocks[ID]) continue;
      removes.push(this.blocks[ID]);
      delete this.blocks[ID];
    }
    for(let ID in this.blocksToCompile) {
      if(!this.items[ID]) continue;
      let neue = this.compileBlock(ID);
      adds.push(neue);
      this.blocks[ID] = neue;
    }
    this.program.blockChangeTransaction(adds, removes);
    this.queued = false;
    this.blocksToCompile = {};
    this.blocksToRemove = {};
  }

  //------------------------------------------------------------------
  // Compiler
  //------------------------------------------------------------------

  inContext(flow:LinearFlow, func: () => void) {
    ReferenceContext.push(flow.context);
    func();
    ReferenceContext.pop();
  }

  compileValue = (compile:CompilationContext, context:ReferenceContext, value:RawValue):Value => {
    let {items} = this;
    if(items[value] && items[value].type === "variable") {
      let found = compile.variables[value];
      if(!found) {
        found = compile.variables[value] = new Reference(context);
      }
      return found;
    }
    return value;
  }

  compileBlock(blockID:string) {
    let {inContext, items, compileValue} = this;
    let item = items[blockID];
    let {name, constraints, type} = item;
    let compile:CompilationContext = {variables: {}};
    let flow:LinearFlow;
    if(type === "block") {
      flow = new LinearFlow(() => []);
    } else if(type === "watch") {
      flow = new WatchFlow(() => []);
    } else {
      flow = new LinearFlow(() => []);
    }
    let {context} = flow;
    for(let constraintID of constraints) {
      let constraint = items[constraintID];
      if(!constraint) continue;

      if(constraint.type === "record") {
        inContext(flow, () => {
          let attrs:any = {};
          for(let [attribute, value] of constraint.attributes) {
            let safeValue = compileValue(compile, context, value);
            let found = attrs[attribute];
            if(!found) {
              found = attrs[attribute] = [];
            }
            found.push(safeValue);
          }
          let record = flow.find(attrs);
          context.equality(record, compileValue(compile, context, constraint.record));
        })
      }
      if(constraint.type === "output") {
        inContext(flow, () => {
          let attrs:any = {};
          for(let [attribute, value] of constraint.attributes) {
            attrs[attribute] = compileValue(compile, context, value);
          }
          let record = flow.record(attrs);
          context.equality(record, compileValue(compile, context, constraint.record));
        })
      }
    }
    let block = (this.program as any)[`_${type}`](name, flow);
    if(type === "watch" && item.watcher) {
      let func = watcherFunctions[item.watcher];
      if(!func) {
        console.error("No such watcher function registered: " + item.watcher);
      } else {
        this.program.asDiffs(func);
      }
    }
    console.log("Compiled: ", block);
    return block;
  }

  //------------------------------------------------------------------
  // Compile item extraction via watch blocks
  //------------------------------------------------------------------

  setup() {
    let {program:me} = this;

    me.watch("get blocks", ({find, record}) => {
      let block = find("eve/compiler/block");
      let {constraint, name, type} = block;
      return [
        record({block, constraint, name, type})
      ]
    })

    me.asObjects<{block:string, name:string, constraint:string, type:string}>(({adds, removes}) => {
      let {items} = this;
      for(let key in adds) {
        let {block, name, constraint, type} = adds[key];
        let found = items[block];
        if(!found) {
          found = items[block] = {type, name, constraints: []};
        }
        found.name = name;
        if(found.constraints.indexOf(constraint) === -1) {
          found.constraints.push(constraint);
        }
        this.queue(block);
      }
      for(let key in removes) {
        let {block, name, constraint} = removes[key];
        let found = items[block];
        if(!found) {
          continue;
        }
        let ix = found.constraints.indexOf(constraint)
        if(ix > -1) {
          found.constraints.splice(ix, 1);
        }
        if(found.constraints.length === 0) {
          delete items[block];
        }
        this.queue(block);
      }
    })

    me.watch("get watcher property", ({find, record}) => {
      let block = find("eve/compiler/block");
      let {watcher} = block;
      return [
        record({block, watcher})
      ]
    })

    me.asObjects<{block:string, watcher:string}>(({adds, removes}) => {
      let {items} = this;
      for(let key in adds) {
        let {block, watcher} = adds[key];
        let found = items[block];
        found.watcher = watcher;
        this.queue(block);
      }
      for(let key in removes) {
        let {block, watcher} = removes[key];
        let found = items[block];
        if(!found) {
          continue;
        }
        found.watcher = undefined;
        this.queue(block);
      }
    })

    me.watch("get variables", ({find, record}) => {
      let variable = find("eve/compiler/var");
      return [
        record({variable})
      ]
    })

    me.asObjects<{variable:string}>(({adds, removes}) => {
      let {items} = this;
      for(let key in adds) {
        let {variable} = adds[key];
        items[variable] = {type: "variable"};
      }
    })

    me.watch("get records", ({find, record}) => {
      let compilerRecord = find("eve/compiler/record");
      let block = find("eve/compiler/block", {constraint: compilerRecord});
      let {record:id, attribute} = compilerRecord;
      return [
        record({block, id:compilerRecord, record:id, attribute:attribute.attribute, value:attribute.value})
      ]
    })

    me.asObjects<{block:string, id:string, record:string, attribute:string, value:RawValue}>(({adds, removes}) => {
      let {items} = this;
      for(let key in adds) {
        let {block, id, record, attribute, value} = adds[key];
        let found = items[id];
        if(!found) {
          found = items[id] = {type: "record", attributes: [], record: record};
        }
        found.attributes.push([attribute, value]);
        this.queue(block);
      }
      for(let key in removes) {
        let {block, id, record, attribute, value} = removes[key];
        let found = items[id];
        if(!found) { continue; }

        found.attributes.filter(([a, v]:RawValue[]) => a !== attribute || v !== value);
        if(found.attributes.length === 0) {
          delete items[id];
        }
        this.queue(block);
      }
    })

    me.watch("get outputs", ({find, record}) => {
      let compilerRecord = find("eve/compiler/output");
      let block = find("eve/compiler/block", {constraint: compilerRecord});
      let {record:id, attribute} = compilerRecord;
      return [
        record({block, id:compilerRecord, record:id, attribute:attribute.attribute, value:attribute.value})
      ]
    })

    me.asObjects<{block:string, id:string, record:string, attribute:string, value:RawValue}>(({adds, removes}) => {
      let {items} = this;
      for(let key in adds) {
        let {block, id, record, attribute, value} = adds[key];
        let found = items[id];
        if(!found) {
          found = items[id] = {type: "output", attributes: [], record: record};
        }
        found.attributes.push([attribute, value]);
        this.queue(block);
      }
      for(let key in removes) {
        let {block, id, record, attribute, value} = removes[key];
        let found = items[id];
        if(!found) { continue; }

        found.attributes.filter(([a, v]:RawValue[]) => a !== attribute || v !== value);
        if(found.attributes.length === 0) {
          delete items[id];
        }
        this.queue(block);
      }
    })

  }
}

Watcher.register("compiler", CompilerWatcher);
