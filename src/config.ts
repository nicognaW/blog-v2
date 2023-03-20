import type { Site, SocialObjects } from "@types";

export const SITE: Site = {
  website: "https://astro-paper.pages.dev/",
  author: "NICO",
  desc: "NK's website",
  title: "NK's website",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerPage: 3,
};

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 864,
  height: 92,
};

export const SOCIALS: SocialObjects = [
  {
    name: "Github",
    href: "https://github.com/nicognaw",
    linkTitle: ` ${SITE.title} on Github`,
    active: true,
  },
];
