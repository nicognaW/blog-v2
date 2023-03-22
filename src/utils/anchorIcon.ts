import type { Root, PhrasingContent } from "mdast";
import { visit } from "unist-util-visit";

const anchor = () => `
<svg xmlns="http://www.w3.org/2000/svg" width="21" height="21" viewBox="0 0 21 21" fill="#00A67E">
    <path d="M14.102 10.31l2.273-2.882a2.71 2.71 0 00-4.255-3.356L9.847 6.954a2.71 2.71 0 00.587 3.907s.886 1.57-.188 2.827A5.342 5.342 0 017.78 5.324l2.272-2.882a5.342 5.342 0 018.39 6.616l-2.273 2.882a5.32 5.32 0 01-1.943 1.537s.17-.966.125-1.585c-.042-.594-.325-1.49-.325-1.49l.076-.092z"/>
    <path d="M8.385 16.743l2.273-2.882a2.71 2.71 0 00-.45-3.806c-.076-.189-.993-1.462.093-2.914a5.342 5.342 0 012.424 8.35l-2.272 2.882a5.342 5.342 0 01-8.39-6.616l2.273-2.882a5.32 5.32 0 011.938-1.536c-.23 1.112-.175 2.104.13 3.166L4.13 13.387a2.71 2.71 0 004.255 3.356z"/>
</svg>
`;

const encodeAnchor = (anchor: string) =>
  encodeURI(Buffer.from(anchor).toString("base64"));

export default function remarkAnchorIcon() {
  return (tree: any) => {
    visit(tree, "heading", node => {
      if (!node.depth) return;
      if (node.depth !== 2 && node.depth !== 3) return;
      if (
        !node.children ||
        !node.children.length ||
        !node.children[0].value ||
        typeof node.children[0].value !== "string"
      )
        return;
      const title = node.children[0].value
        .toLowerCase()
        .replace(/ /g, "-")
        .replace(/[^\u4e00-\u9fa5a-z0-9-]/g, "");
      if (title === "目录") return;
      node.children = [createAnchor(title), ...node.children];
    });
  };
}

export const createAnchor = (id: string): PhrasingContent => ({
  type: "link",
  url: `#${id}`,
  children: [
    {
      type: "image",
      alt: "#",
      url: `data:image/svg+xml;base64,${encodeAnchor(anchor())}`,
      data: {
        hProperties: {
          className: "anchor-icon",
        },
      },
    },
  ],
  data: {
    hProperties: {
      className: "anchor-icon",
    },
  },
});
