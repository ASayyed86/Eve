import {Program} from "./runtime/dsl2";

let prog = new Program("test");
prog.attach("html");

prog
  .block("simple block", ({find, record, lib}) => {
    let person = find("P");
    let potato = find("potato");
    let nameElem;
    return [
      nameElem = record("html/element", {tagname: "span", text: person.name}),
      record("html/element", {tagname: "section", potato}).add("child" + "ren", nameElem)
    ]
  });

prog.test(0, [
  [2, "tag", "html/element"],
  [2, "tagname", "div"],
  [2, "children", 3],
  [2, "sort", 1],

  [3, "tag", "html/element"],
  [3, "tagname", "span"],
  [3, "text", "Woo hoo!"],
  [3, "style", 4],

  [4, "color", "red"],
  [4, "background", "pink"],

  [5, "tag", "html/element"],
  [5, "tagname", "div"],
  [5, "style", 6],
  [5, "children", 7],
  [5, "sort", 3],

  [6, "border", "3px solid green"],

  [7, "tag", "html/element"],
  [7, "tagname", "span"],
  [7, "text", "meep moop"],
]);

prog.test(1, [
  [3, "style", 4, 0, -1]
]);

prog.test(2, [
  [3, "style", 4, 0, 1],
  [4, "font-size", "4em"],
  [4, "background", "pink", 0, -1]
]);

prog.test(3, [
  [8, "tag", "html/element"],
  [8, "tagname", "div"],
  [8, "style", 4],
  [8, "text", "Jeff (from accounting)"],
  [8, "sort", 0]
]);

