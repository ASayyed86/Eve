//--------------------------------------------------------------------
// Editor
//--------------------------------------------------------------------

import {Watcher, Program, RawMap, RawValue, RawEAV, forwardDiffs} from "../watchers/watcher";
import {CompilerWatcher} from "../watchers/compiler";
import {v4 as uuid} from "node-uuid";

//--------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------

function isRawValue(x:any): x is RawValue {
  return x !== undefined && (typeof x === "string" || typeof x === "number");
}

function isRawValueArray(x:any): x is RawValue[] {
  if(x && x.constructor === Array) {
    for(let value of x) {
      if(!isRawValue(value)) return false;
    }
    return true;
  }
  return false;
}

function isRawEAVArray(x:any): x is RawEAV[] {
  if(x && x.constructor === Array) {
    for(let value of x) {
      if(!isRawValueArray(value)) return false;
      if(value.length !== 3) return false;
    }
    return true;
  }
  return false;
}


interface Attrs extends RawMap<RawValue|RawValue[]|RawEAV[]|RawEAV[][]> {}
function appendAsEAVs(eavs:any[], record: Attrs, id = `|${uuid()}`) {
  for(let attr in record) {
    let value = record[attr];
    if(isRawValue(value)) {
      eavs.push([id, attr, value]);

    } else if(isRawValueArray(value)) {
      // We have a set of scalars
      for(let val of value) eavs.push([id, attr, val]);

    } else if(isRawEAVArray(value)) {
      // We have a single nested sub-object (i.e. a set of EAVs).
      let childEavs = value;
      let [childId] = childEavs[0];
      eavs.push([id, attr, childId]);
      for(let childEav of childEavs) eavs.push(childEav);

    } else {
      // We have a set of nested sub-objects.
      for(let childEavs of value) {
        let [childId] = childEavs[0];
        eavs.push([id, attr, childId]);
        for(let childEav of childEavs) eavs.push(childEav);
      }
    }
  }

  return eavs;
}

//--------------------------------------------------------------------
// Fixture constants
//--------------------------------------------------------------------

const EDITOR_ID = `|${uuid()}`;
const STYLE_ID = `|${uuid()}`;

const TAG_MARINA_ID = `|${uuid()}`;
const TAG_MARINARA_ID = `|${uuid()}`;
const BLOCK_PPL_W_BOATS_ID = `|${uuid()}`;
const NODE_PERSON_ID = `|${uuid()}`;
const NODE_BOAT_ID = `|${uuid()}`;
const BLOCK_BOAT_TYPES_ID = `|${uuid()}`;
const FRAME_PPL_W_BOATS_QUERY_ID = `|${uuid()}`;

const PERSON_1_ID = `|${uuid()}`;
const PERSON_2_ID = `|${uuid()}`;
const PERSON_3_ID = `|${uuid()}`;
const BOAT_1_ID = `|${uuid()}`;
const BOAT_2_ID = `|${uuid()}`;
const BOAT_3_ID = `|${uuid()}`;

const DOCK_1_ID = `|${uuid()}`;

//--------------------------------------------------------------------
// Watcher
//--------------------------------------------------------------------

class EditorWatcher extends Watcher {
  editor:Program;


  setup() {
    this.editor = this.createEditor();
    let {editor, program} = this;

    this.program
      .watch("Export all tags.", ({find, lib:{string}, not, record}) => {
        let rec = find();
        let {tag} = rec;
        not(() => string.index_of(tag, "editor/") == 0);
        return [
          record("client_tag", {"client_tag": tag})
        ];
      })
      .asDiffs(forwardDiffs(editor))

      .watch("Send the editor attributes on records matching the tag.", ({find, lookup, record}) => {
        let attributes_from_tag = find("editor/attributes-from-tag");
        let {query_tag} = attributes_from_tag;
        let rec = find({tag: query_tag});
        let {attribute} = lookup(rec);

        return [
          attributes_from_tag.add({tag: "editor/attributes-from-tag", query_tag, attribute})
        ];
      })
      .asDiffs(forwardDiffs(editor))

      .watch("Send the editor attributes on records matching the parent's children.", ({find, lookup, record}) => {
        let attributes_from_parent = find("editor/attributes-from-parent");
        let {parent_tag, parent_attribute} = attributes_from_parent;
        let parent = find({tag: parent_tag});
        let {attribute:par_attr, value:rec} = lookup(parent);
        parent_attribute == par_attr;
        let {attribute} = lookup(rec);

        return [
          attributes_from_parent.add({tag: "editor/attributes-from-parent", parent_tag, parent_attribute, attribute})
        ];
      })
      .asDiffs(forwardDiffs(editor))

      // @FIXME: Quick hack to get things running.
      .watch("Attributes with a | are ids.", ({find, lookup, lib:{string}, record}) => {
        let attributes_from_tag = find("editor/attributes-from-tag");
        let {query_tag} = attributes_from_tag;
        let rec = find({tag: query_tag});
        let {attribute, value} = lookup(rec);
        string.index_of(value, "|");
        return [
          attributes_from_tag.add("record_attribute", attribute)
        ];
      })
      .asDiffs(forwardDiffs(editor));

    this.initEditor();
    this.fixtureClient();
    this.fixtureEditor();
  }

  createEditor() {
    let editor = new Program("Editor");
    editor.attach("ui");
    editor.attach("shape");

    let compiler = editor.attach("compiler") as CompilerWatcher;
    compiler.injectInto(this.program);
    compiler.registerWatcherFunction("send-to-editor", forwardDiffs(editor, "send-to-editor", true));


    editor
      .block("All html elements add their tags as classes", ({find, lib:{string}, record}) => {
        let element = find("html/element");
        element.tag != "html/element"
        let klass = string.replace(element.tag, "/", "-");
        return [
          element.add("class", klass)
        ];
      });

    //--------------------------------------------------------------------
    // Root UI
    //--------------------------------------------------------------------

    editor
      .block("Draw the root editor view.", ({find, record}) => {
        let editor = find("editor/root");

        return [
          record("editor/view", "ui/row", {editor}).add("children", [
            record("editor/nav", "ui/column", {editor, sort: 0}),
            record("editor/main", "ui/column", {editor, sort: 1}).add("children", [
              record("ui/row", {editor, sort: 0, class: "editor-block-header"}).add("children", [
                record("editor/block/description", "ui/column", {editor}),
                record("editor/block/storyboard", "ui/row", {editor})
              ]),
              record("ui/row", "editor/block/content", {editor, sort: 1})
            ])
          ])
        ];
      });

    //--------------------------------------------------------------------
    // Navigation
    //--------------------------------------------------------------------

    editor
      .block("Populate the nav bar with the program's block tags.", ({find, record}) => {
        let nav = find("editor/nav");
        let tag = nav.editor.block.nav_tag;
        return [
          nav.add("children", [
            record("editor/nav/tag", "ui/column", {editor: nav.editor, sort: tag.name, nav_tag: tag}).add("children", [
              record("ui/text", {sort: 0, text: tag.name})
            ])
          ])
        ];
      })

      .block("Populate nav tags with the blocks that have them.", ({find, choose, record}) => {
        let tag = find("editor/nav/tag");
        let block = tag.editor.block;
        block.nav_tag == tag.nav_tag;

        let [name] = choose(
          () => block.name,
          () => "Untitled Block"
        );

        return [
          tag.add("children", [
            record("editor/nav/block", "ui/text", {editor: tag.editor, nav_tag: tag.nav_tag, block, text: name, sort: name})
          ])
        ];
      });

    //--------------------------------------------------------------------
    // Block Description
    //--------------------------------------------------------------------

    editor
      .block("Populate the block description for the active block.", ({find, choose, record}) => {
        let description = find("editor/block/description");
        let active_block = description.editor.active_block;

        let [name] = choose(() => active_block.name, () => "Untitled Block");
        let [text] = choose(() => active_block.description, () => "");

        return [
          description.add("children", [
            record("ui/text", {sort: 0, text: name, class: "editor-block-title"}),
            record("ui/text", {sort: 1, text})
          ])
        ];
      });

    //--------------------------------------------------------------------
    // Block Frames
    //--------------------------------------------------------------------

    editor
      .block("Populate the block storyboard for the active block.", ({find, record}) => {
        let storyboard = find("editor/block/storyboard");
        let {editor} = storyboard;
        let {active_block} = editor;
        let frame = active_block.storyboard;
        return [
          storyboard.add("children", [
            record("editor/block/frame", "ui/column", {editor, sort: frame.sort, frame}).add("children", [
              record("ui/text", {text: frame.type})
            ])
          ])
        ];
      })

      .block("Mark the active frame.", ({find}) => {
        let editor = find("editor/root");
        let {active_frame:frame} = editor;
        let frame_elem = find("editor/block/frame", {frame});
        return [
          frame_elem.add("class", "active")
        ];
      })

      .commit("Clicking a frame activates it", ({find}) => {
        let frame_elem = find("editor/block/frame");
        find("html/event/click", {element: frame_elem});
        let {frame, editor} = frame_elem;
        return [
          editor.remove("active_frame").add("active_frame", frame)
        ];
      })

      .block("Add a new frame button to the storyboard.", ({find, record}) => {
        let storyboard = find("editor/block/storyboard");
        let {editor} = storyboard;
        let {active_block} = editor;
        return [
          storyboard.add("children", [
            record("editor/new-frame", "editor/block/frame", "ui/column", {editor, sort: Infinity})
          ])
        ];
      })

      .commit("Clicking the new frame button opens it", ({find}) => {
        let new_frame = find("editor/new-frame");
        find("html/event/click", "html/direct-target", {element: new_frame});
        return [
          new_frame.add("open", "true")
        ];
      })

      .block("When the new frame is open, display a list of editor types to choose from.", ({find, record}) => {
        let new_frame = find("editor/new-frame", {open: "true"});
        let {editor} = new_frame;
        return [
          new_frame.add("children", [
            record("editor/new-frame/type", "ui/button", {editor, text: "Query", type: "query", class: "flat"}),
            record("editor/new-frame/type", "ui/button", {editor, text: "Data", type: "data", class: "flat"}),
          ])
        ];
      })

      .commit("Clicking a new frame type adds a frame of that type and closes the new frame button.", ({find, gather, choose, record}) => {
        let new_frame_type = find("editor/new-frame/type");
        find("html/event/click", "html/direct-target", {element: new_frame_type});
        let {type, editor} = new_frame_type;
        let new_frame = find("editor/new-frame", {editor});

        let {active_block:block} = editor;
        let [ix] = choose(
          () => gather(block.storyboard).per(block).count() + 1,
          () => 1
        );

        return [
          new_frame.remove("open"),
          block.add("storyboard", [
            record("editor/frame", {block, type, sort: ix})
          ])
        ];
      });

    this.attachQueryEditor(editor);

    return editor;
  }

  attachQueryEditor(editor:Program) {
    editor.block("When the active frame is a query, inject the query editor UI.", ({find, union, record}) => {
      let content = find("editor/block/content");
      let {editor} = content;
      editor.active_frame.type == "query";

      return [
        content.add("children", [
          record("editor/block/query-tree", "ui/column", {editor}),
          record("editor/block/query-canvas", {editor})
        ])
      ];
    });

    //--------------------------------------------------------------------
    // Block Query Tree
    //--------------------------------------------------------------------

    editor
      .block("Compute the label and color of query nodes.", ({find, choose, lib:{string}}) => {
        let node = find("editor/query-node");
        let {name, sort} = node;
        let label = string.uppercase(string.get(name, 1));
        let color = choose(() => find("node-color", {ix: sort}).color, () => "gray");
        return [node.add({label, color})];
      })

      .block("Populate the block query for the active block.", ({find, union, record}) => {
        let query_elem = find("editor/block/query-tree");
        let {editor} = query_elem;
        let {active_frame} = editor;
        active_frame.type == "query";
        let {node} = active_frame;

        let [main_pattern] = union(
          () => node.query_tag,
          () => node.parent_attribute
        );

        return [
          record("editor/query/node", "ui/row", {editor, sort: node.sort, node, frame: active_frame}).add("children", [
            record("editor/query/hex", "shape/hexagon", {side: 21, thickness: 2, border: "#AAA", background: "white", sort: 0, frame: active_frame, node}).add("content", [
              record("ui/text", {text: node.label, style: record({color: node.color})})
            ]),
            record("editor/query/pattern", "ui/column", {sort: 1, frame: active_frame, node}).add("children", [
              record("ui/text", {sort: 0, text: main_pattern, class: "editor-query-tag"}),
            ])
          ])
        ];
      })

      .block("Query root nodes are children of the query.", ({find, union, record}) => {
        let query_elem = find("editor/block/query-tree");
        let root_node = find("editor/query/node", {editor: query_elem.editor, node: find("editor/root-node")});
        return [
          query_elem.add("children", root_node)
        ];
      })

      .block("Non-root nodes are children of their parent node.", ({find, union, record}) => {
        let query_elem = find("editor/block/query-tree");
        let subnode = find("editor/query/node", {editor: query_elem.editor});
        let {node} = subnode;
        let {parent_node} = node;
        let parent_pattern = find("editor/query/pattern", {node: parent_node});
        return [
          parent_pattern.add("children", [
            record("ui/column", {node, sort: 4, class: "editor-query-subnode"}).add("children", subnode)
          ])
        ];
      })

      .block("Query nodes with attributes display them as a tree in the pattern.", ({find, record}) => {
        let query_pattern = find("editor/query/pattern");
        let {node} = query_pattern;
        let {query_field} = node;
        return [
          query_pattern.add("children", [
            record("ui/column", {node,  sort: 2}).add("children", [
              record("ui/row", {node, query_field, sort: query_field, class: "editor-query-field"}).add("children", [
                record("ui/text", {text: query_field}),
                record("editor/query/delete-field", "ui/button", {node, query_field, icon: "close-round"})
              ])
            ])
          ])
        ];
      })

      .commit("Clicking a delete field button removes its attribute from the pattern", ({find, record}) => {
        let delete_field = find("editor/query/delete-field");
        let click = find("html/event/click", {element: delete_field});
        let {node, query_field} = delete_field;
        return [
          node.remove("query_field", query_field)
        ];
      })

      .commit("Clicking a query hex opens the add attribute menu", ({find, record}) => {
        let query_hex = find("editor/query/hex");
        let click = find("html/event/click", {element: query_hex});
        let {node} = query_hex;
        let query_node = find("editor/query/node", {node});
        return [
          query_node.add("new-attribute", "true")
        ];
      })
      .watch("If a query node is adding an attribute, request attributes matching its tag from the client.", ({find, record}) => {
        let query_node = find("editor/query/node", {"new-attribute": "true"});
        let {node} = query_node;
        return [record("editor/attributes-from-tag", {query_tag: node.query_tag})];
      })
      .asDiffs(forwardDiffs(this.program))

      .watch("If a query node is adding an attribute, request attributes matching its position in the hierarchy.", ({find, record}) => {
        let query_node = find("editor/query/node", {"new-attribute": "true"});
        let {node} = query_node;
        return [record("editor/attributes-from-parent", {parent_tag: node.parent_node.query_tag, parent_attribute: node.parent_attribute})];
      })
      .asDiffs(forwardDiffs(this.program))


      .block("When a query node is in the new attribute state, show all the attributes matching its tag", ({find, choose, record}) => {
        let query_node = find("editor/query/node", {"new-attribute": "true"});
        let {node} = query_node;
        let query_pattern = find("editor/query/pattern");
        query_node.children == query_pattern;
        let [attribute] = choose(
          () => {
            let {attribute} = find("editor/attributes-from-tag", {query_tag: node.query_tag});
            return [attribute];
          },
          () => {
            let {attribute} = find("editor/attributes-from-parent", {parent_tag: node.parent_node.query_tag, parent_attribute: node.parent_attribute});
            return [attribute];
          },
        );

        return [
          query_pattern.add("children", [
            record("ui/column", {node,  sort: 3}).add("children", [
              record("editor/query/node/new-attribute", "ui/text", {text: attribute, sort: attribute, attribute, node})
            ])
          ])
        ];
      })

      .commit("Clicking a new attribute in a query node adds it.", ({find, not, record}) => {
        let new_attribute = find("editor/query/node/new-attribute");
        let click = find("html/event/click", {element: new_attribute});
        let {node, attribute} = new_attribute;
        let query_node = find("editor/query/node", {node});
        not(() => find("editor/attributes-from-tag", {query_tag: node.query_tag, record_attribute: attribute}))
        return [
          node.add("query_field", attribute),
          query_node.remove("new-attribute")
        ];
      })

    // @FIXME: Getting multiple subnodes if we have existing attrs but not vice versa
      .commit("Clicking a new record attribute in a query node adds it as a sub-node.", ({find, gather, choose, record}) => {
        let new_attribute = find("editor/query/node/new-attribute");
        let click = find("html/event/click", {element: new_attribute});
        let {node, attribute} = new_attribute;
        let query_node = find("editor/query/node", {node});

        find("editor/attributes-from-tag", {query_tag: node.query_tag, record_attribute: attribute});

        let frame = query_node.frame;
        let [ix] = choose(() => {
          return [gather(frame.node).count() + 1];
        }, () => 1);

        return [
          node.add("query_subnode", attribute),
          query_node.frame.add("node", [
            record("editor/query-node", "editor/subnode", {
              type: "join",
              sort: ix,
              name: attribute,
              parent_attribute: attribute,
              parent_node: node,
            })
          ]),
          query_node.remove("new-attribute")
        ];
      })

      .block("The query always has a new node button", ({find, record}) => {
        let query_elem = find("editor/block/query-tree");
        let {editor} = query_elem;
        let {active_frame} = editor;

        return [
          query_elem.add("children", [
            record("ui/row", "editor/query/new-node", {sort: 9999, frame: active_frame}).add("children", [
              record("shape/hexagon", {side: 21, thickness: 2, border: "#AAA", background: "white", class: "editor-query-hex"}).add("content", [
                record("ui/text", {text: "+", style: record({color: "#AAA", "font-weight": 500})})
              ]),
            ])
          ])
        ];
      })

      .commit("Clicking on the new node button opens it.", ({find, not, record}) => {
        let new_node_button = find("editor/query/new-node");
        not(() => new_node_button.open == "true");
        let click = find("html/event/click", {element: new_node_button});
        return [
          new_node_button.add("open", "true")
        ];
      })

      .block("When the new node button is open, display a list of the clients tags.", ({find, record}) => {
        let new_node_button = find("editor/query/new-node", {open: "true"});
        let tag = find("client_tag").client_tag;
        return [
          new_node_button.add("children", [
            record("ui/column", {sort: 1}).add("children", [
              record("ui/text", "editor/query/new-node/tag", {text: tag, sort: tag, client_tag: tag, new_node_button})
            ])
          ])
        ];
      })

      .commit("Clicking on a new node tag adds it as a node to the query.", ({find, gather, choose, record}) => {
        let new_node_tag = find("editor/query/new-node/tag");
        let {new_node_button} = new_node_tag;
        let {client_tag} = new_node_tag;
        let {frame} = new_node_button;
        let click = find("html/event/click", {element: new_node_tag});

        // @FIXME: dependents of aggregates are busted due to stratification (?).
        // If we try to use ix directly for scanning we get no result.
        // If we try to use it after an expression for scanning we get no filtering at all.
        // Luckily in this case we needed it in a choose, which seems to stratify correctly.

        // @FIXME: Aggregates in chooses don't filter adequately without context.
        // We work around it for now by providing enough context in the choose branch for the aggregate to use.

        let [ix] = choose(() => {
          return [gather(frame.node).count() + 1];
        }, () => 1);

        return [
          new_node_button.remove("open"),
          frame.add("node", [
            record("editor/query-node", "editor/root-node", {
              type: "join",
              sort: ix,
              name: client_tag,
              query_tag: client_tag,
              // query_field: "name"
            })
          ])
        ];
      });

    //--------------------------------------------------------------------
    // Block Query Canvas
    //--------------------------------------------------------------------

    editor
      .commit("Molecules start with a seed of 1", ({find, not}) => {
        let molecule = find("editor/molecule");
        not(() => molecule.seed);
        return [molecule.add("seed", 1)];
      })
      .block("Find a potential location for new molecules", ({find, lib:{random, math}, record}) => {
        let molecule = find("editor/molecule");
        let {seed, atom} = molecule;

        let x = math.round(random.number(`${molecule} ${seed} x`) * 10);
        let y = math.round(random.number(`${molecule} ${seed} y`) * 5);

        return [
          molecule.add({x, y}) // , positioned: "true"
        ];
      })
      .commit("A molecule with positioned false and a low enough seed should try to reposition.", ({find, record}) => {
        let molecule = find("editor/molecule", {positioned: "false"});
        molecule.seed < 4;
        return [
          molecule.remove("positioned").remove("seed").add("seed", molecule.seed + 1),
        ];
      })

      .block("Determine the suitability of a potential molecule position by colliding it's footprint with existing cells.", ({find, choose, not}) => {
        let molecule = find("editor/molecule");
        molecule.seed < 4;
        let delay = find("someone-is-maybe-positioned");
        //not(() => molecule.positioned);
        let [positioned] = choose(
          () => {
            let {skirt} = molecule;
            let other = find("editor/molecule");
            other.seed < 4;
            molecule.frame == other.frame;
            molecule.generation >= other.generation;
            molecule != other;
            let {atom:other_atom} = other;
            other_atom.x == skirt.x;
            other_atom.y == skirt.y;
            return "false";
          },
          () => {
            return "true";
          }
        );
        return [
          molecule.add("positioned", positioned)
        ];
      })

      .block("create a skirt around unpositioned molecules.", ({find, not, lib:{math}, record}) => {
        let molecule = find("editor/molecule");
        //not(() => molecule.positioned == "true");
        let {atom} = molecule;
        let {ix} = find("range");
        let {x, y} = find("spiral", {row: math.mod(atom.y, 2), ix});
        return [molecule.add("skirt", [
          record("editor/molecule/skirt", {x: atom.x + x, y: atom.y + y})
        ])];
      })

      .block("Sort atoms by id.", ({find, gather, record}) => {
        let molecule = find("editor/molecule");
        let {atom} = molecule;
        let ix = gather(atom).per(molecule).sort();
        return [
          atom.add("sort", ix)
        ];
      })

      .commit("When we first see a molecule, mark its generation.", ({find, not, choose, gather}) => {
        let molecule = find("editor/molecule");
        not(() => molecule.generation);
        let [generation] = choose(
          () => {
            let existing = find("editor/molecule");
            existing.generation;
            return gather(existing).count();
          },
          () => 1
        );

        return [molecule.add("generation", generation)];
      })

      .block("DEBUG: Sort molecules by id.", ({find, gather, record}) => {
        let molecule = find("editor/molecule");
        let ix = gather(molecule.generation, molecule).sort();
        return [
          molecule.add("sort", ix)
        ];
      })

      // .block("DEBUG: Show the molecules.", ({find, record}) => {
      //   let molecule = find("editor/molecule");
      //   let {atom} = molecule;
      //   return [
      //     record("ui/text", {sort: `${molecule.sort}${atom.sort}`, text: `${molecule.sort} | ${atom.node.label} ${atom.sort}`})
      //   ];
      // })

      .block("Compute atom positions from their sort.", ({find, lib:{math}, record}) => {
        let molecule = find("editor/molecule");
        let {atom, x:mol_x, y:mol_y} = molecule;
        let {x, y} = find("spiral", {row: math.mod(molecule.y, 2), ix: atom.sort});
        return [
          atom.add({x: mol_x + x, y: mol_y + y}),
          record("someone-is-maybe-positioned")
        ];
      })

      .block("Draw molecules as hex grids of atoms.", ({find, record}) => {
        let canvas_elem = find("editor/block/query-canvas");
        let {editor} = canvas_elem;
        editor.active_frame.type == "query";
        let molecule = find("editor/molecule", {editor, positioned: "true"});
        let {atom} = molecule;

        let side = 30;
        let gap = 3;

        return [
          canvas_elem.add({tag: "shape/hex-grid", side, gap}),
          canvas_elem.add("cell", [
            // record("editor/molecule/grid", "shape/hex-grid", {x: molecule.x, y: molecule.y, side, gap}).add("cell", [
            record("shape/hexagon", "editor/atom/cell", {atom, molecule, side, x: atom.x, y: atom.y, background: "white", thickness: 2, border: "#ccc"}).add("content", [
                record("ui/text", {atom, molecule, text: `${atom.node.label} ${molecule.sort}`, style: record({color: atom.node.color})})
              ])
            // ])
          ])
        ];
      })

      .commit("Clicking on an atom cell opens it's molecule.", ({find, not}) => {
        let atom_cell = find("editor/atom/cell");
        find("html/event/click", {element: atom_cell});
        let {molecule} = atom_cell;
        not(() => molecule.open == "true");
        return [
          molecule.add("open", "true")
        ];
      })

      .commit("Clicking on an atom cell closes any currently open molecules.", ({find, not}) => {
        let atom_cell = find("editor/atom/cell");
        find("html/event/click", {element: atom_cell});
        let {molecule} = atom_cell;
        let {editor} = molecule;
        let other_molecule = find("editor/molecule", {editor, open: "true"});

        return [
          other_molecule.remove("open")
        ];
      })


      .block("Show molecule infobox when open.", ({find, lookup, record}) => {
        let molecule = find("editor/molecule", {open: "true"});
        let {atom, editor} = molecule;
        let canvas_elem = find("editor/block/query-canvas", {editor});
        let {field} = atom;
        let {attribute, value} = lookup(field);
        return [
          canvas_elem.add("children", [
            record("ui/column", "editor/molecule/infobox", {sort: molecule.sort, molecule}).add("children", [
              record("ui/text", {text: `Molecule ${molecule.sort}`}),
              record("ui/row", {sort: atom.sort, molecule, atom}).add("children", [
                record("ui/text", {sort: 0, text: `${atom.node.name} {`}),
                record("ui/text", {sort: atom.sort, text: ` ${attribute}: ${value} `}),
                record("ui/text", {sort: Infinity, text: `}`}),
              ])
            ])
          ])
        ];
      });

    //--------------------------------------------------------------------
    // Molecule generation
    //--------------------------------------------------------------------

    editor.block("Create a set of molecules for the active frame's query.", ({find, record}) => {
      let editor = find("editor/root");
      let {active_frame} = editor;
      active_frame.type == "query";
      let {node} = active_frame;
      node.tag == "editor/root-node";

      return [
        record("editor/molecule/watch", "eve/compiler/block", {editor, name: "Create molecules", type: "watch", watcher: "send-to-editor"}).add("constraint", [
          record("editor/atom/record", "eve/compiler/record", {node, record: record("editor/atom/entity", "eve/compiler/var", {node})}),
        ]),
      ];
    })

    editor.block("Embed subnodes.", ({find, record}) => {
      let molecule_watch = find("editor/molecule/watch");
      let {editor} = molecule_watch;
      let {active_frame} = editor;
      let {node} = active_frame;
      let {parent_node, parent_attribute} = node;

      let parent_record = find("editor/atom/record", {node: parent_node});

      let record_var;
      return [
        record_var = record("editor/atom/entity", "eve/compiler/var", {node}),
        parent_record.add("attribute", record({tag: "eve/compiler/av", attribute: parent_attribute, value: record_var})),

        molecule_watch.add("constraint", [
          record("editor/atom/record", "eve/compiler/record", {node, record: record_var})
        ])
      ];
    })


    editor.block("Attach node query tags to their atom records.", ({find, record}) => {
      let atom_record = find("editor/atom/record");
      let {node} = atom_record;
      let {query_tag} = node;

      return [
        atom_record.add("attribute", record({tag: "eve/compiler/av", attribute: "tag", value: query_tag})),
      ];
    })

    editor.block("Attach node fields to their atom records.", ({find, record}) => {
      let atom_record = find("editor/atom/record");
      let {node} = atom_record;
      let {query_field} = node;

      return [
        atom_record.add("attribute", [
          record({
            tag: "eve/compiler/av",
            attribute: query_field,
            value: record("editor/atom/field", "eve/compiler/var", {node, query_field})
          })
        ])
      ];
    })

    editor.block("Output a molecule for each root node.", ({find, record}) => {
      let molecule_watch = find("editor/molecule/watch");
      let {editor} = molecule_watch;
      let node = find("editor/root-node");
      let {active_frame} = editor;
      active_frame.node == node;

      let entity_var = find("editor/atom/entity", {node});

      let molecule_var;
      return [
        molecule_var = record("editor/molecule/output_var", "eve/compiler/var", {node}),
        molecule_watch.add("constraint", [
          record("editor/molecule/output", "eve/compiler/output", {molecule_watch, node, record: molecule_var}).add("attribute", [
            record("eve/compiler/av", {attribute: "tag", value: "editor/molecule"}),
            record("eve/compiler/av", {attribute: "editor", value: editor}),
            record("eve/compiler/av", {attribute: "frame", value: active_frame}),
            record("eve/compiler/av", {attribute: "node", value: node}),
            record("eve/compiler/av", {attribute: "root_atom_record", value: entity_var}),
          ])
        ]),
      ];
    });

    editor.block("Attach the root atom to molecules.", ({find, record}) => {
      let molecule_output = find("editor/molecule/output");
      let {molecule_watch} = molecule_output;
      let {node} = molecule_output;

      let atom_var;
      return [
        atom_var = record("editor/atom/output_var", "eve/compiler/var", {node}),
        molecule_watch.add("constraint", [
          record("editor/atom/output", "eve/compiler/output", {node, molecule_output, record: atom_var})
        ]),
        molecule_output.add("parent_node", node),

      ];
    });

    editor.block("Attach subnode atoms to molecules.", ({find, record}) => {
      let molecule_output = find("editor/molecule/output");
      let {molecule_watch, parent_node} = molecule_output;
      let {node} = find("editor/query/node");
      node.parent_node == parent_node;

      let atom_var;
      return [
        atom_var = record("editor/atom/output_var", "eve/compiler/var", {node}),
        molecule_watch.add("constraint", [
          record("editor/atom/output", "eve/compiler/output", {node, molecule_output, record: atom_var})
        ]),
        molecule_output.add("parent_node", node),
      ];
    });

    editor.block("Fill vital atom output attributes and attach them to their molecule output.", ({find, record}) => {
      let atom_output = find("editor/atom/output");
      let {molecule_output, node} = atom_output;
      let entity_var = find("editor/atom/entity", {node});
      return [
        atom_output.add("attribute", [
          record("eve/compiler/av", {attribute: "tag", value: "editor/atom"}),
          record("eve/compiler/av", {attribute: "node", value: node}),
          record("eve/compiler/av", {attribute: "molecule", value: molecule_output.record}),
          record("eve/compiler/av", {attribute: "record", value: entity_var}),
        ]),

        molecule_output.add("attribute", [
          record("eve/compiler/av", "eve/compiler/attribute/non-identity", {attribute: "atom", value: atom_output.record}),
        ]),
      ];
    })

    // @FIXME: What about root node identity?
    editor.block("Attach node query fields to their atom outputs.", ({find, record}) => {
      let atom_field_var = find("editor/atom/field");
      let {node, query_field} = atom_field_var;
      let atom_output = find("editor/atom/output", {node});
      let {molecule_output} = atom_output;
      let {molecule_watch} = molecule_output;

      let field_var;
      return [
        field_var = record("editor/atom/field/output_var", "eve/compiler/var", {node, root_node: molecule_output.node}),
        molecule_watch.add("constraint", [
          record("editor/atom/field/output", "eve/compiler/output", {node, record: field_var}).add("attribute", [
            record("eve/compiler/av", {attribute: query_field, value: atom_field_var})
          ])
        ]),
        atom_output.add("attribute", [
          record("eve/compiler/av", "eve/compiler/attribute/non-identity", {attribute: "field", value: field_var}),
        ])
      ];
    })
  }

  initEditor() {
    //--------------------------------------------------------------------
    // Kick it off
    //--------------------------------------------------------------------
    let input:RawEAV[] = [
      [EDITOR_ID, "tag", "editor/root"],
      [STYLE_ID, "tag", "html/element"],
      [STYLE_ID, "tagname", "link"],
      [STYLE_ID, "rel", "stylesheet"],
      [STYLE_ID, "href", "assets/css/editor.css"],
    ];

    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 1, x: 0, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 2, x: 1, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 3, x: 1, y: 1});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 4, x: 0, y: 1});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 5, x: -1, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 6, x: 0, y: -1});
    appendAsEAVs(input, {tag: "spiral", row: 1, ix: 7, x: 1, y: -1});

    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 1, x: 0, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 2, x: 1, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 3, x: 0, y: 1});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 4, x: -1, y: 1});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 5, x: -1, y: 0});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 6, x: -1, y: -1});
    appendAsEAVs(input, {tag: "spiral", row: 0, ix: 7, x: 0, y: -1});

    appendAsEAVs(input, {tag: "range", ix: 0});
    appendAsEAVs(input, {tag: "range", ix: 1});
    appendAsEAVs(input, {tag: "range", ix: 2});
    appendAsEAVs(input, {tag: "range", ix: 3});
    appendAsEAVs(input, {tag: "range", ix: 4});
    appendAsEAVs(input, {tag: "range", ix: 5});
    appendAsEAVs(input, {tag: "range", ix: 6});
    appendAsEAVs(input, {tag: "range", ix: 7});
    appendAsEAVs(input, {tag: "range", ix: 8});
    appendAsEAVs(input, {tag: "range", ix: 9});
    appendAsEAVs(input, {tag: "range", ix: 10});

    // appendAsEAVs(input, {tag: "node-color", ix: 1, color: "red"});
    // appendAsEAVs(input, {tag: "node-color", ix: 2, color: "orange"});
    // appendAsEAVs(input, {tag: "node-color", ix: 3, color: "yellow"});
    // appendAsEAVs(input, {tag: "node-color", ix: 4, color: "green"});
    // appendAsEAVs(input, {tag: "node-color", ix: 5, color: "blue"});
    // appendAsEAVs(input, {tag: "node-color", ix: 6, color: "indigo"});
    // appendAsEAVs(input, {tag: "node-color", ix: 7, color: "violet"});
    // appendAsEAVs(input, {tag: "node-color", ix: 8, color: "light gray"});
    // appendAsEAVs(input, {tag: "node-color", ix: 9, color: "dark gray"});

    appendAsEAVs(input, {tag: "node-color", ix: 1, color: "#9926ea"});
    appendAsEAVs(input, {tag: "node-color", ix: 2, color: "#6c86ff"});
    appendAsEAVs(input, {tag: "node-color", ix: 3, color: "red"});
    appendAsEAVs(input, {tag: "node-color", ix: 4, color: "orange"});
    appendAsEAVs(input, {tag: "node-color", ix: 5, color: "green"});
    appendAsEAVs(input, {tag: "node-color", ix: 6, color: "indigo"});


    this.editor.inputEavs(input);
  }

  fixtureEditor() {
    let fixture:RawEAV[] = [
      [EDITOR_ID, "block", BLOCK_PPL_W_BOATS_ID],
      [EDITOR_ID, "block", BLOCK_BOAT_TYPES_ID],
      [EDITOR_ID, "active_block",  BLOCK_PPL_W_BOATS_ID],
      [EDITOR_ID, "active_frame",  FRAME_PPL_W_BOATS_QUERY_ID]
    ];

    appendAsEAVs(fixture, {name: "Marina"}, TAG_MARINA_ID);
    appendAsEAVs(fixture, {name: "Marinara"}, TAG_MARINARA_ID);

    appendAsEAVs(fixture, {
      tag: "editor/block",
      nav_tag: [TAG_MARINA_ID, TAG_MARINARA_ID],
      name: "People with boats",
      description: `Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.`,
      storyboard: [
        appendAsEAVs([], {
          tag: "editor/frame",
          type: "query",
          sort: 1,
          node: [
            // appendAsEAVs([], {
            //   tag: ["editor/query-node"],
            //   type: "join",
            //   sort: 1,
            //   label: "P",
            //   color: "#6c86ff",
            //   query_tag: "person",
            //   query_field: ["name", "age", "boat"],

            //   join: [
            //     appendAsEAVs([], {
            //       attribute: "boat",
            //       other_node: NODE_BOAT_ID
            //     })
            //   ]
            // }, NODE_PERSON_ID),
            // appendAsEAVs([], {
            //   tag: "editor/query-node",
            //   type: "join",
            //   sort: 1,
            //   label: "B",
            //   color: "#9926ea",
            //   query_tag: "boat",
            //   query_field: ["name", "type"]
            // }, NODE_BOAT_ID)
          ]
        }, FRAME_PPL_W_BOATS_QUERY_ID)
      ]
    }, BLOCK_PPL_W_BOATS_ID);

    // appendAsEAVs(fixture, {
    //   tag: "editor/molecule",
    //   editor: EDITOR_ID,
    //   node: NODE_PERSON_ID,
    //   atom: [
    //     appendAsEAVs([], {tag: "editor/atom", sort: 1, node: NODE_PERSON_ID, record: PERSON_1_ID}),
    //     appendAsEAVs([], {tag: "editor/atom", sort: 2, node: NODE_BOAT_ID, record: BOAT_1_ID}),
    //     appendAsEAVs([], {tag: "editor/atom", sort: 3, node: NODE_BOAT_ID, record: BOAT_3_ID}),
    //   ]
    // }),

    appendAsEAVs(fixture, {
      tag: "editor/block",
      nav_tag: TAG_MARINA_ID,
      name: "Boat types",
      description: `It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of labels, as opposed to using 'Content here, content here', making it look like readable English.`
    }, BLOCK_BOAT_TYPES_ID);

    this.editor.inputEavs(fixture);
  }

  fixtureClient() {

    let fixture:RawEAV[] = [];
    appendAsEAVs(fixture, {tag: "person", name: "Josh", boat: [BOAT_1_ID, BOAT_3_ID], age: 23}, PERSON_1_ID);
    appendAsEAVs(fixture, {tag: "person", name: "Rafe", boat: BOAT_1_ID, age: 43}, PERSON_2_ID);
    appendAsEAVs(fixture, {tag: "person", name: "Lola", boat: BOAT_2_ID, age: 19}, PERSON_3_ID);

    appendAsEAVs(fixture, {tag: "boat", name: "Boaty Mcboatface", type: "yacht", dock: DOCK_1_ID}, BOAT_1_ID);
    appendAsEAVs(fixture, {tag: "boat", name: "H.M. Surf", type: "dinghy", dock: DOCK_1_ID}, BOAT_2_ID);
    appendAsEAVs(fixture, {tag: "boat", name: "No Life Raft", type: "dinghy", dock: DOCK_1_ID}, BOAT_3_ID);

    appendAsEAVs(fixture, {tag: "dock", name: "Marinara Marina of Michigan", state: "MI"}, DOCK_1_ID);
    this.program.inputEavs(fixture);
  }
}

Watcher.register("editor", EditorWatcher);
