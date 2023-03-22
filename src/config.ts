import type { Site, SocialObjects } from "@types";

export const SITE: Site = {
  website: "https://astro-paper.pages.dev/",
  author: "NICO",
  desc: "NK's Insights",
  title: "NK's Insights",
  ogImage: "logo.jpeg",
  lightAndDarkMode: true,
  postPerPage: 3,
};

export const LOGO_IMAGE = {
  enable: true,
  svg: false,
  width: 96,
  height: 96,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/nicognaw",
    linkTitle: ` ${SITE.title} on Github`,
    active: true,
  },
  {
    name: "Mail",
    href: "mailto:nicognaw@outlook.com",
    linkTitle: `Send an email to ${SITE.title}`,
    active: true,
  },
  {
    name: "BiliBili",
    href: "mailto:nicognaw@outlook.com",
    linkTitle: `Send an email to ${SITE.title}`,
    active: true,
  },
];
