//--------------------------------------------------------------------
// Flappy
//--------------------------------------------------------------------

import {Program} from "../runtime/dsl2";
import {RawMap, RawValue, RawEAV} from "../watchers/watcher";
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
function appendAsEAVs(eavs:any[], record: Attrs, id = uuid()) {
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
// Program
//--------------------------------------------------------------------

let prog = new Program("editor");
prog.attach("ui");
prog.attach("shape");

//--------------------------------------------------------------------
// Root UI
//--------------------------------------------------------------------

prog
  .block("Draw the root editor view.", ({find, record}) => {
    let editor = find("editor/root");

    return [
      record("editor/view", "ui/row", {editor, class: "editor-view"}).add("children", [
        record("editor/nav", "ui/column", {editor, class: "editor-nav"}),
        record("ui/column", {editor, class: "editor-main"}).add("children", [
          record("ui/row", {editor, sort: 0, class: "editor-block-header"}).add("children", [
            record("editor/block/description", "ui/column", {editor, class: "editor-block-description"}),
            record("editor/block/storyboard", "ui/row", {editor, class: "editor-block-storyboard"})
          ]),
          record("ui/row", {editor, sort: 1, class: "editor-block-content"}).add("children", [
            record("editor/block/query", "ui/column", {editor, class: "editor-block-query"}),
            record("editor/block/canvas", {editor, class: "editor-block-canvas"})
          ])
        ])
      ])
    ];
  })

//--------------------------------------------------------------------
// Navigation
//--------------------------------------------------------------------

  .block("Populate the nav bar with the program's block tags.", ({find, record}) => {
    let nav = find("editor/nav");
    let tag = nav.editor.block.nav_tag;
    return [
      nav.add("children", [
        record("editor/nav/tag", "ui/column", {editor: nav.editor, sort: tag.name, nav_tag: tag, class: "editor-nav-tag"}).add("children", [
          record("ui/text", {sort: 0, text: tag.name})
        ])
      ])
    ];
  })

  .block("Populate nav tags with the blocks that have them.", ({find, choose, record}) => {
    let tagElem = find("editor/nav/tag");
    let block = tagElem.editor.block;
    block.nav_tag == tagElem.nav_tag;

    let [name] = choose(
      () => block.name,
      () => "Untitled Block"
    );

    return [
      tagElem.add("children", [
        record("editor/nav/block", "ui/text", {editor: tagElem.editor, nav_tag: tagElem.nav_tag, block, text: name, sort: name, class: "editor-nav-block"})
      ])
    ];
  });

//--------------------------------------------------------------------
// Block Description
//--------------------------------------------------------------------

prog.block("Populate the block description for the active block.", ({find, choose, record}) => {
  let descriptionElem = find("editor/block/description");
  let active_block = descriptionElem.editor.active_block;

  let [name] = choose(
    () => active_block.name,
    () => "Untitled Block"
  );

  let [description] = choose(
    () => active_block.description,
    () => ""
  );

  return [
    descriptionElem.add("children", [
      record("ui/text", {sort: 0, text: name, class: "editor-block-title"}),
      record("ui/text", {sort: 1, text: description})
    ])
  ];
});

//--------------------------------------------------------------------
// Block Frames
//--------------------------------------------------------------------

prog.block("Populate the block storyboard for the active block.", ({find, record}) => {
  let storyboardElem = find("editor/block/storyboard");
  let {editor} = storyboardElem;
  let {active_block} = editor;
  let frame = active_block.storyboard;
  return [
    storyboardElem.add("children", [
      record("editor/block/frame", "ui/column", {editor, sort: frame.sort, frame, class: "editor-block-frame"}).add("children", [
        record("ui/text", {text: frame.type})
      ])
    ])
  ];
});

prog.block("Add a 'new frame' button to the storyboard.", ({find, record}) => {
  let storyboardElem = find("editor/block/storyboard");
  let {editor} = storyboardElem;
  let {active_block} = editor;
  return [
    storyboardElem.add("children", [
      record("editor/new-frame", "ui/column", {editor, sort: Infinity, class: ["editor-block-frame", "editor-new-frame"]})
    ])
  ];
});

//--------------------------------------------------------------------
// Block Query
//--------------------------------------------------------------------

prog.block("Populate the block query for the active block.", ({find, record}) => {
  let queryElem = find("editor/block/query");
  let {editor} = queryElem;
  let {active_frame} = editor;
  let {node} = active_frame;
  return [
    queryElem.add("children", [
      record("ui/row", "editor/block/query-node", {editor, sort: node.sort, node, class: "editor-block-query-node"}).add("children", [
        record("shape/hexagon", {side: 21, thickness: 2, border: "#AAA", background: "white", sort: 0, frame: active_frame, node, class: "editor-block-query-hex"}).add("content", [
          record("ui/text", {text: node.label, style: record({color: node.color})})
        ]),
        record("ui/column", {sort: 1, frame: active_frame, node, class: "editor-block-query-pattern"}).add("children", [
          record("ui/text", {sort: 0, text: node.queryTag, class: "editor-query-tag"}),
          record("ui/text", {sort: node.queryField, text: node.queryField, class: "editor-query-field"})
        ])
      ])
    ])
  ];
});

//--------------------------------------------------------------------
// Block Canvas
//--------------------------------------------------------------------

prog.block("Draw molecules as hex grids of atoms.", ({find, record, lib:{random, math}}) => {
  let canvas_elem = find("editor/block/canvas");
  let {editor} = canvas_elem;
  let molecule = find("editor/molecule", {editor});

  let {atom} = molecule;
  let {x, y} = find("spiral", {ix: atom.sort})

  let molecule_x = math.round(random.number(`${molecule} x`) * 10);
  let molecule_y = math.round(random.number(`${molecule} y`) * 6);

  return [
    canvas_elem.add({
      tag: "shape/hex-grid",
      side: 30,
      gap: 3
    }),
    canvas_elem.add("cell", [
      record("editor/molecule/grid", "shape/hex-grid", {x: molecule_x, y: molecule_y, side: 30, gap: 3}).add("cell", [
        record({atom, x, y, background: "white", thickness: 2, border: "#ccc"}).add("content", [
          record("ui/text", {text: atom.node.label, style: record({color: atom.node.color})})
        ])
      ])
    ])
  ];
});

//--------------------------------------------------------------------
// Kick it off
//--------------------------------------------------------------------

const EDITOR_ID = uuid();
const STYLE_ID = uuid();

prog.inputEavs([
  [EDITOR_ID, "tag", "editor/root"],
  [STYLE_ID, "tag", "html/element"],
  [STYLE_ID, "tagname", "link"],
  [STYLE_ID, "rel", "stylesheet"],
  [STYLE_ID, "href", "assets/css/editor.css"],
]);

//--------------------------------------------------------------------
// Data Fixture
//--------------------------------------------------------------------

const TAG_MARINA_ID = uuid();
const TAG_MARINARA_ID = uuid();
const BLOCK_PPL_W_BOATS_ID = uuid();
const NODE_PERSON_ID = uuid();
const NODE_BOAT_ID = uuid();
const BLOCK_BOAT_TYPES_ID = uuid();
const FRAME_PPL_W_BOATS_QUERY_ID = uuid();

let fixture:RawEAV[] = [
  [EDITOR_ID, "block", BLOCK_PPL_W_BOATS_ID],
  [EDITOR_ID, "block", BLOCK_BOAT_TYPES_ID],
  [EDITOR_ID, "active_block",  BLOCK_PPL_W_BOATS_ID],
  [EDITOR_ID, "active_frame",  FRAME_PPL_W_BOATS_QUERY_ID]
];

// We can't do range yet.
appendAsEAVs(fixture, {
  tag: "range",
  ix: [1, 3, 4]
});

const PERSON_1_ID = uuid();
const PERSON_2_ID = uuid();
const PERSON_3_ID = uuid();
const BOAT_1_ID = uuid();
const BOAT_2_ID = uuid();
const BOAT_3_ID = uuid();

appendAsEAVs(fixture, {tag: "person", name: "josh", boat: [BOAT_1_ID, BOAT_3_ID], age: 23}, PERSON_1_ID);
appendAsEAVs(fixture, {tag: "person", name: "rafe", boat: BOAT_1_ID, age: 43}, PERSON_2_ID);
appendAsEAVs(fixture, {tag: "person", name: "lola", boat: BOAT_2_ID, age: 19}, PERSON_3_ID);

appendAsEAVs(fixture, {tag: "boat", name: "boaty mcboatface", type: "yacht"}, BOAT_1_ID);
appendAsEAVs(fixture, {tag: "boat", name: "H.M. Surf", type: "dinghy"}, BOAT_2_ID);
appendAsEAVs(fixture, {tag: "boat", name: "No Life Raft", type: "dinghy"}, BOAT_3_ID);


appendAsEAVs(fixture, {
  name: "Marina"
}, TAG_MARINA_ID);

appendAsEAVs(fixture, {
  name: "Marinara"
}, TAG_MARINARA_ID);


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
        appendAsEAVs([], {
          tag: ["editor/query-node"],
          type: "join",
          sort: 1,
          label: "P",
          color: "#6c86ff",
          queryTag: "person",
          queryField: ["name", "age", "boat"],

          join: [
            appendAsEAVs([], {
              attribute: "boat",
              other_node: NODE_BOAT_ID
            })
          ]
        }, NODE_PERSON_ID),
        appendAsEAVs([], {
          tag: "editor/query-node",
          type: "join",
          sort: 1,
          label: "B",
          color: "#9926ea",
          queryTag: "boat",
          queryField: ["name", "type"]
        }, NODE_BOAT_ID)
      ]
    }, FRAME_PPL_W_BOATS_QUERY_ID)
  ]
}, BLOCK_PPL_W_BOATS_ID);

appendAsEAVs(fixture, {
  tag: "editor/molecule",
  editor: EDITOR_ID,
  node: NODE_PERSON_ID,
  atom: [
    appendAsEAVs([], {tag: "editor/atom", sort: 1, node: NODE_PERSON_ID, record: PERSON_1_ID}),
    appendAsEAVs([], {tag: "editor/atom", sort: 2, node: NODE_BOAT_ID, record: BOAT_1_ID}),
    appendAsEAVs([], {tag: "editor/atom", sort: 3, node: NODE_BOAT_ID, record: BOAT_3_ID}),
  ]
}),

appendAsEAVs(fixture, {
  tag: "editor/molecule",
  editor: EDITOR_ID,
  node: NODE_PERSON_ID,
  atom: [
    appendAsEAVs([], {tag: "editor/atom", sort: 1, node: NODE_PERSON_ID, record: PERSON_2_ID}),
    appendAsEAVs([], {tag: "editor/atom", sort: 2, node: NODE_BOAT_ID, record: BOAT_1_ID}),
  ]
})

appendAsEAVs(fixture, {
  tag: "editor/molecule",
  editor: EDITOR_ID,
  node: NODE_PERSON_ID,
  atom: [
    appendAsEAVs([], {tag: "editor/atom", sort: 1, node: NODE_PERSON_ID, record: PERSON_3_ID}),
    appendAsEAVs([], {tag: "editor/atom", sort: 2, node: NODE_BOAT_ID, record: BOAT_2_ID}),
  ]
})

appendAsEAVs(fixture, {
  tag: "editor/block",
  nav_tag: TAG_MARINA_ID,
  name: "Boat types",
  description: `It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of labels, as opposed to using 'Content here, content here', making it look like readable English.`
}, BLOCK_BOAT_TYPES_ID);

appendAsEAVs(fixture, {tag: "spiral", ix: 1, x: 0, y: 0});
appendAsEAVs(fixture, {tag: "spiral", ix: 2, x: 1, y: 0});
appendAsEAVs(fixture, {tag: "spiral", ix: 3, x: 0, y: 1});
appendAsEAVs(fixture, {tag: "spiral", ix: 4, x: -1, y: 1});
appendAsEAVs(fixture, {tag: "spiral", ix: 5, x: -1, y: 0});
appendAsEAVs(fixture, {tag: "spiral", ix: 6, x: -1, y: -1});
appendAsEAVs(fixture, {tag: "spiral", ix: 7, x: 0, y: -1});

prog.inputEavs(fixture);

console.log(prog);
