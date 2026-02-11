import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Easy-langent",
  description: "简单的lang框架智能体开发实践教程",
  base: '/easy-langent/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '开始学习', link: '/guide/前言' }
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '前言', link: '/guide/前言' },
          { text: '第一章 LangChain与LangGraph框架认知', link: '/guide/chapter1' }
        ]
      },
      {
        text: 'LangChain组件与实战',
        items: [
          { text: '第二章 LangChain核心组件实操', link: '/guide/chapter2' },
          { text: '第三章 LangChain进阶组件实操', link: '/guide/chapter3' },
          { text: '第四章 LangChain应用级系统设计与RAG实践', link: '/guide/chapter4' },
          { text: '第五章 课程中期综合实践', link: '/guide/chapter5' }
        ]
      },
      {
        text: 'LangGraph组件与实战',
        items: [
          { text: '第六章 LangGraph基础', link: '/guide/chapter6' },
          { text: '第七章 LangGraph进阶', link: '/guide/chapter7' },
          { text: '第八章 综合实战：谁是卧底', link: '/guide/chapter8' }
        ]
      },
      {
        text: '结语',
        items: [
          { text: '结语', link: '/guide/结语' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/easy-langent' }
    ]
  }
})
