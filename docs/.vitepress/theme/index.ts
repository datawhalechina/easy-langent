import DefaultTheme from 'vitepress/theme'
import './custom.css'
import type { Theme } from 'vitepress'
import 'viewerjs/dist/viewer.min.css';
import imageViewer from 'vitepress-plugin-image-viewer';
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue';
import { useRoute } from 'vitepress';
import { h } from 'vue';


export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        // 注册全局组件（可选）
        app.component('vImageViewer', vImageViewer);
    },
    setup() {
        const route = useRoute();
        // 启用插件
        imageViewer(route);
    },
} satisfies Theme
