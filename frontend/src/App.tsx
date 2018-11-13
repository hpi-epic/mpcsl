import { Layout } from 'antd';
import React, { Component } from 'react';
import colors from './colors';

const { Header, Content, Footer }  = Layout;

import './App.css';

class App extends Component {
  public render() {
    return (
      <Layout className='layout'>
        <Header>
          hjlasd
        </Header>
        <Content style={{ background: colors.contentBackground }}>
          content
        </Content>
        <Footer>
          Made by HPI
        </Footer>
      </Layout>
    );
  }
}

export default App;
